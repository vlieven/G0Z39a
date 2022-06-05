import warnings
from abc import ABC, abstractmethod
from string import Template
from typing import Sequence

import pandas as pd
from neo4j import Query, Record
from neo4j.exceptions import ClientError

from .. import Connection


class BaseEmbedding(ABC):
    def __init__(self, projection_name: str):
        self._projection_name: str = projection_name

    @property
    def projection_name(self) -> str:
        return self._projection_name

    def create_projection(
        self, connection: Connection, force: bool = False
    ) -> Sequence[Record]:
        if force:
            self.drop_projection(connection)

        return self._create_projection(connection)

    def drop_projection(self, connection: Connection) -> Sequence[Record]:
        template: Template = Template(
            """
                CALL gds.graph.drop('$projection')
                """
        )

        query: Query = Query(template.substitute(projection=self.projection_name))

        try:
            return connection.query(query)
        except ClientError as e:
            warnings.warn(str(e))
            return []

    @abstractmethod
    def _create_projection(self, connection: Connection) -> Sequence[Record]:
        raise NotImplementedError

    def estimate_memory(
        self, connection: Connection, *, embedding_dimension: int
    ) -> Sequence[Record]:
        template: Template = Template(
            """
            CALL gds.fastRP.stream.estimate('$projection', {embeddingDimension: $dimension})
            YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
            RETURN nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
            """
        )

        query: Query = Query(
            template.substitute(
                projection=self.projection_name, dimension=str(embedding_dimension)
            )
        )
        return connection.query(query)


class CountyEmbedding(BaseEmbedding):
    def __init__(self, name: str = "counties", random_seed: int = 42):
        super().__init__(projection_name=name)
        self.random_seed: int = random_seed

    def _create_projection(self, connection: Connection) -> Sequence[Record]:
        """Creates a native projection in Neo4J"""
        template: Template = Template(
            """
            CALL gds.graph.project(
                '$projection',            
                {
                    County: { 
                        properties: {
                            census: {defaultValue: 0},
                            pop_under_5: {defaultValue: 0},
                            pop_5_to_17: {defaultValue: 0},
                            pop_18_to_65: {defaultValue: 0},
                            pop_plus_65: {defaultValue: 0},
                            is_metro: {defaultValue: 0},
                            svi_a: {defaultValue: 0},
                            svi_b: {defaultValue: 0},
                            svi_c: {defaultValue: 0},
                            svi_d: {defaultValue: 0}
                        }
                    }
                },         
                {
                    IS_NEAR: {
                        properties: 'weight',
                        orientation: 'UNDIRECTED'
                    }
                }
            )
            YIELD
                graphName AS graph, 
                nodeProjection, 
                nodeCount AS nodes, 
                relationshipProjection, 
                relationshipCount AS rels
            RETURN 
                graph,
                nodeProjection.County AS countyProjection,
                nodes,
                rels
            """
        )

        query: Query = Query(template.substitute(projection=self.projection_name))

        try:
            return connection.query(query)
        except ClientError as e:
            warnings.warn(str(e))
            return []

    def generate_embedding(
        self,
        connection: Connection,
        *,
        embedding_dimension: int = 64,
        weight2: float = 0.0,
        weight3: float = 0.5,
        weight4: float = 1.0,
        normalization: float = -0.5,
        property_ratio: float = 0.0,
        self_influence: float = 0.0,
    ) -> Sequence[Record]:
        template: Template = Template(
            """
            CALL gds.fastRP.stream('$projection', 
                {
                    randomSeed: $seed,
                    embeddingDimension: $dimension,
                    iterationWeights: [0.0, $weight2, $weight3, $weight4],
                    nodeSelfInfluence: $self_influence,
                    normalizationStrength: $normalization,
                    relationshipWeightProperty: 'weight',
                    propertyRatio: $property_ratio,
                    featureProperties: [
                        'census',
                        'pop_under_5',
                        'pop_5_to_17',
                        'pop_18_to_65',
                        'pop_plus_65',
                        'is_metro',
                        'svi_a',
                        'svi_b',
                        'svi_c',
                        'svi_d'
                    ]
                }
            )
            YIELD nodeId, embedding
            """
        )

        query: Query = Query(
            template.substitute(
                seed=str(self.random_seed),
                projection=self.projection_name,
                dimension=str(embedding_dimension),
                weight2=str(weight2),
                weight3=str(weight3),
                weight4=str(weight4),
                normalization=str(normalization),
                property_ratio=str(property_ratio),
                self_influence=str(self_influence),
            )
        )
        return connection.query(query)

    @classmethod
    def node_id_to_fips_mapping(cls, connection: Connection) -> Sequence[Record]:
        query: Query = Query(
            """
            MATCH (c:County) 
            RETURN ID(c) AS nodeId, c.fips as fips
            """
        )

        return connection.query(query)

    def load_dataframe(
        self,
        connection: Connection,
        *,
        embedding_dimension: int = 64,
        weight2: float = 0.0,
        weight3: float = 0.5,
        weight4: float = 1.0,
        normalization: float = -0.5,
        property_ratio: float = 0.0,
        self_influence: float = 0.0,
    ) -> pd.DataFrame:
        idx: str = "nodeId"

        embeddings: pd.DataFrame = pd.DataFrame(
            [
                dict(row)
                for row in self.generate_embedding(
                    connection,
                    embedding_dimension=embedding_dimension,
                    weight2=weight2,
                    weight3=weight3,
                    weight4=weight4,
                    normalization=normalization,
                    property_ratio=property_ratio,
                    self_influence=self_influence,
                )
            ]
        ).set_index(idx)

        nodes: pd.DataFrame = pd.DataFrame(
            [dict(row) for row in self.node_id_to_fips_mapping(connection)]
        ).set_index(idx)

        result: pd.DataFrame = nodes.join(embeddings, how="inner").set_index("fips")
        return pd.DataFrame(
            result["embedding"].to_list(),
            index=result.index,
            columns=[f"emb_{_}" for _ in range(embedding_dimension)],
        )
