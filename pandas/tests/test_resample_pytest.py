from datetime import datetime, timedelta

import pytest
import numpy as np

import pandas as pd
import pandas.util.testing as tm
from pandas import (Series, DataFrame, notnull)
from pandas.tseries.frequencies import to_offset, MONTHS
from pandas.core.indexes.datetimes import date_range
from pandas.tseries.offsets import Minute
from pandas.core.indexes.period import period_range, PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.resample import TimeGrouper


def assert_series_or_frame_equal(result, expected):
    if isinstance(result, Series):
        return tm.assert_series_equal(result, expected)
    else:
        return tm.assert_frame_equal(result, expected)


class Base(object):
    """
    base class for resampling testing
    - pandas_obj fixture to generate Series/DataFrame of each index type using
      static default index and values
    - call self.create_random_value_series() to generate series of each
      index type
    """

    _index_fixture_start = datetime(2005, 1, 1)
    _index_fixture_end = datetime(2005, 1, 10)
    _index_fixture_freq = 'D'

    def create_index(self, *args, **kwargs):
        """ return the _index_factory created using the args, kwargs """
        factory = self._index_factory()
        return factory(*args, **kwargs)

    def create_random_value_series(self, start, end, freq='D'):
        # replace _simple_ts() and _simple_pts() functions, dispatches on the
        # subclass-specific index type
        rng = self.create_index(start, end, freq=freq)
        return Series(np.random.randn(len(rng)), index=rng)

    @pytest.fixture(scope='class')
    def index(self):
        return self.create_index(self._index_fixture_start,
                                 self._index_fixture_end,
                                 freq=self._index_fixture_freq)

    @pytest.fixture(params=[Series, DataFrame], scope='class')
    def pandas_obj(self, request, index):
        if request.param == Series:
            return Series(np.arange(len(index)), index=index,
                          name=self._series_fixture_name)
        if request.param == DataFrame:
            return DataFrame({'value': np.arange(len(index))}, index=index)

    @pytest.mark.parametrize('pandas_obj', [Series, DataFrame], indirect=True)
    @pytest.mark.parametrize('freq, freq_mult', [('2D', 2)])
    # yields 2 test cases for each subclass (except when overridden)
    def test_asfreq_downsample(self, pandas_obj, freq, freq_mult):
        obj = pandas_obj
        result = obj.resample(freq).asfreq()
        expected = obj.reindex(obj.index.take(np.arange(0, len(obj.index),
                                                        freq_mult)))
        expected.index.freq = to_offset(freq)
        assert_series_or_frame_equal(result, expected)

    @pytest.mark.parametrize('pandas_obj', [DataFrame], indirect=True)
    @pytest.mark.parametrize('agg_arg', ['mean', {'value': 'mean'}, ['mean']])
    # yields 3 test cases for each subclass
    def test_resample_loffset_arg_type(self, pandas_obj, agg_arg):
        # GH 13218, 15002
        df = pandas_obj
        expected_means = [df.values[i:i + 2].mean()
                          for i in range(0, len(df.values), 2)]
        expected_index = self.create_index(df.index[0],
                                           periods=len(df.index) / 2,
                                           freq='2D')

        # loffset coreces PeriodIndex to DateTimeIndex
        if isinstance(expected_index, PeriodIndex):
            expected_index = expected_index.to_timestamp()

        expected_index += timedelta(hours=2)
        expected = DataFrame({'value': expected_means}, index=expected_index)

        result_agg = df.resample('2D', loffset='2H').agg(agg_arg)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result_how = df.resample('2D', how=agg_arg, loffset='2H')

        if isinstance(agg_arg, list):
            expected.columns = pd.MultiIndex.from_tuples([('value', 'mean')])

        # GH 13022, 7687 - TODO: fix resample w/ TimedeltaIndex
        if isinstance(expected.index, TimedeltaIndex):
            with pytest.raises(AssertionError):
                tm.assert_frame_equal(result_agg, expected)
                tm.assert_frame_equal(result_how, expected)
        else:
            tm.assert_frame_equal(result_agg, expected)
            tm.assert_frame_equal(result_how, expected)


class TestDatetimeIndex(Base):
    _index_factory = lambda x: date_range
    _series_fixture_name = 'dti'

    @pytest.mark.parametrize('func', ['add', 'mean', 'prod', 'ohlc', 'min',
                                      'max', 'var'])
    @pytest.mark.parametrize('grouper', [TimeGrouper(Minute(5)),
                                         TimeGrouper(Minute(5),
                                                     closed='right',
                                                     label='right')])
    # 14 test cases
    def test_custom_grouper(self, func, grouper):
        index = self.create_index(freq='Min', start=datetime(2005, 1, 1),
                                  end=datetime(2005, 1, 10))
        s = Series(np.array([1] * len(index)), index=index, dtype='int64')
        b = grouper
        g = s.groupby(b)
        # check all cython functions work
        g._cython_agg_general(func)

        tm.assert_almost_equal(g.ngroups, 2593)
        assert notnull(g.mean()).all()

        df = DataFrame(np.random.rand(len(index), 10),
                       index=index, dtype='float64')
        r = df.groupby(b).agg(np.sum)
        assert len(r.columns) == 10
        assert len(r.index) == 2593

        if grouper == TimeGrouper(Minute(5), closed='right', label='right'):
            # construct expected val
            arr = [1] + [5] * 2592
            idx = index[0:-1:5]
            idx = idx.append(index[-1:])
            expect = Series(arr, index=idx)

            # GH2763 - return in put dtype if we can
            result = g.agg(np.sum)
            tm.assert_series_equal(result, expect)


class TestPeriodIndex(Base):
    _index_factory = lambda x: period_range
    _series_fixture_name = 'pi'

    # explicitly specify to run test on Series & DataFrame
    @pytest.mark.parametrize('pandas_obj', [Series, DataFrame], indirect=True)
    @pytest.mark.parametrize('freq', ['2D'])
    @pytest.mark.parametrize('kind', ['period', None, 'timestamp'])
    # yields 6 test cases
    def test_asfreq_downsample(self, pandas_obj, freq, kind):
        obj = pandas_obj
        expected = obj.reindex(obj.index.take(np.arange(0, len(obj.index), 2)))
        expected.index = expected.index.to_timestamp()
        expected.index.freq = to_offset(freq)

        # this is a bug, this *should* return a PeriodIndex
        # directly
        # GH 12884
        result = obj.resample(freq, kind=kind).asfreq()
        assert_series_or_frame_equal(result, expected)

    # explicitly restrict pandas_obj fixture to return only Series
    @pytest.mark.parametrize('pandas_obj', [Series], indirect=True)
    @pytest.mark.parametrize('freq', ['1H', '1min'])
    # yields 2 test cases
    def test_asfreq_upsample(self, pandas_obj, freq):
        obj = pandas_obj
        new_index = date_range(obj.index[0].to_timestamp(how='start'),
                               (obj.index[-1] + 1).to_timestamp(how='start'),
                               freq=freq,
                               closed='left')
        expected = obj.to_timestamp().reindex(new_index).to_period()
        result = obj.resample(freq).asfreq()
        tm.assert_series_equal(result, expected)

    # using the plain pandas_obj fixture runs test on Series & DataFrame,
    # equivalent to explicitly state:
    # @pytest.mark.parametrize('pandas_obj', [Series, DataFrame],
    #                          indirect=True)
    # yields 2 test cases
    def test_asfreq_fill_value(self, pandas_obj):
        # test for fill value during resampling, issue 3715
        obj = pandas_obj
        new_index = date_range(obj.index[0].to_timestamp(how='start'),
                               (obj.index[-1]).to_timestamp(how='start'),
                               freq='1H')
        expected = obj.to_timestamp().reindex(new_index, fill_value=4.0)
        result = obj.resample('1H', kind='timestamp').asfreq(fill_value=4.0)
        assert_series_or_frame_equal(result, expected)

    # BEFORE: nested for-loops to represent multiple test cases
    #
    # def test_quarterly_upsample(self):
    #     targets = ['D', 'B', 'M']
    #
    #     for month in MONTHS:
    #         ts = _simple_pts('1/1/1990', '12/31/1995', freq='Q-%s' % month)
    #
    #         for targ, conv in product(targets, ['start', 'end']):
    #             result = ts.resample(targ, convention=conv).ffill()
    #             expected = result.to_timestamp(targ, how=conv)
    #             expected = expected.asfreq(targ, 'ffill').to_period()
    #             assert_series_equal(result, expected)

    # AFTER: Parametrized test on loop variables:
    @pytest.mark.parametrize('target', ['D', 'B', 'M'])
    @pytest.mark.parametrize('month', MONTHS)
    @pytest.mark.parametrize('conv', ['start', 'end'])
    # yiels 3 * 12 * 2 = 72 test cases (carthesian product of parameters)
    def test_quarterly_upsample(self, target, month, conv):
        ts = self.create_random_value_series('1/1/1990', '12/31/1995',
                                             freq='Q-%s' % month)
        result = ts.resample(target, convention=conv).ffill()
        expected = result.to_timestamp(target, how=conv)
        expected = expected.asfreq(target, 'ffill').to_period()
        tm.assert_series_equal(result, expected)
