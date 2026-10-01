# This file is part of ts_m1m3_cli.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import unittest

import vcr
from astropy.time import Time, TimeDelta

from lsst.ts.m1m3.cli import detect_earthquake_signals

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "none"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class TestEarthquakes(unittest.IsolatedAsyncioTestCase):
    """Test earthquakes detection"""

    async def test_detect_earthquake_signals(self) -> None:
        t1 = Time("2026-07-04T02:53:00Z")
        t2 = Time("2026-07-04T03:20:00Z")

        with myvcr.use_cassette("detect_earthquake_signals"):
            earthquake_signals = await detect_earthquake_signals(
                t1, t2, chunk_timedelta=TimeDelta(1200, format="sec")
            )

        assert len(earthquake_signals.index) == 78

        assert earthquake_signals["timestamp"][0] == Time("2026-07-04 03:09:51.881999Z")
        assert earthquake_signals["timestamp"][77] == Time("2026-07-04 03:10:03.648519Z")


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}")
    unittest.main()
