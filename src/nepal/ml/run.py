"""
The dataset that we are working with can be considered panel time series
https://www.sktime.org/en/stable/glossary.html#term-Panel-time-series

I want to solve this using a global model:
https://www.business-science.io/code-tools/2021/07/19/modeltime-panel-data.html

After testing, Orbit (by Uber) seems not mature enough for our purposes,
it does not support panel data, en for a single timeseries, the example code resulted
in a numpy error.

Prophet (by Facebook/Meta) is mainly used for data with strong seasonality,
which does not really apply to our case.
"""
