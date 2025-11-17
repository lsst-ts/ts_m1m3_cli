#############
ts_m1m3_cli
#############



.. Provides M1M3 command line tools.

Provides command line tools to extract data from EFD and run analysis of the data.

... correlate-timeseries

Correlates timeseries. Fits data on a smallest residuals.

... compare-two-tma-settings

Compare telescope performance at two TMA settings. Usefull to run analysis of
M1M3 behavior under different settings.

... find-changes

Looks for changes in data. Usefull to detect mostly booleans flags changes.

... m1m3-aav

Fits acceleration and velocity forces. The fitted forces are then computed
real-time and added to M1M3 forces as the TMA slews.

... m1m3-bump-tests-times

Shows M1M3 force actuators (FA) bump test times.

... m1m3-do-bump-tests

Starts M1M3 bump tests.

... m1m3-fcu-stats

Prints M1M3 FCU statistics.

... m1m3-fe-outliers

Looks for the M1M3 FA following error outliers. Usefull for determinging which
force actuators contribute to overall large following error.
