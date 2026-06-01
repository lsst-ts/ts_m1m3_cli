# This file is part of ts_m1m3_cli.
#
# Developed for the LSST Telescope and Site.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import asyncio
import sys

import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient

from lsst.ts.m1m3.utils import DurationTime, HPForces


async def get_data(client: EfdClient, field: str, t_start: Time, t_end: Time) -> pd.DataFrame:
    """
    Returns data from Efd

    Parameters
    ----------
    client : `EfdClient`
    field : `str`
    t_start : `Time`
    t_end : `Time`

    Returns
    -------
    data : `pd.DataFrame`
    """
    print(f"Analyzing {field} from {t_start} to {t_end}")
    data = await client.select_time_series("lsst.sal.MTM1M3.hardpointActuatorData", field, t_start, t_end)
    return data


async def main() -> None:
    now = Time.now()

    parser = argparse.ArgumentParser(description="Calculate HP excesses for a given definition of excess")
    parser.add_argument("--efd", default="usdf_efd", help="EFD name. Defaults to usdf_efd.")
    parser.add_argument(
        "start_time",
        type=DurationTime(now),
        default=now - TimeDelta(3600, format="sec"),
        help="Start time for excess check in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "end_time",
        type=DurationTime(now),
        default=now,
        help="End time for excess check in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "hp_id",
        choices=[str(i) for i in range(1, 7)] + ["all"] + ["odd"] + ["even"],
        default=["all"],
        action="append",
        help="Hardpoint identifier [1-6], all, odd or even.",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=1,
        help="Reference period (in seconds) in which an 'excess' is defined",
    )
    parser.add_argument(
        "--delta-f-threshold",
        type=float,
        default=200,
        help="Minimum increment of force (in N) defining an excess (in delta_t)",
    )
    parser.add_argument(
        "--time-gap-threshold",
        type=str,
        default="1s",
        help="Minimum time between events to consider a grouped excess. 1s is a good value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save output to a given filename",
    )
    args = parser.parse_args()
    client = EfdClient(args.efd)

    indexes = []

    for hp_index in args.hp_id:
        try:
            indexes.append(int(hp_index) - 1)
        except ValueError:
            if hp_index == "all":
                indexes = list(range(1, 7))
            elif hp_index == "even":
                indexes += list(range(2, 7, 2))
            elif hp_index == "odd":
                indexes += list(range(1, 7, 2))

    indexes = sorted(set(indexes))

    hp_excesses = []

    sampling_freq = 50  # Hz

    start_t, end_t = DurationTime.pair(args.start_time, args.end_time)

    for hp_index in indexes:
        df = await get_data(client, f"measuredForce{hp_index - 1}", start_t, end_t)
        time_gap_threshold = TimeDelta(args.time_gap_threshold)
        hpf = HPForces(df, sampling_freq, time_gap_threshold)
        excess = hpf.calculate_excesses(hp_index, args.delta_t, args.delta_f_threshold)
        excess["hp"] = hp_index
        hp_excesses.append(excess)

    event_summary = pd.concat(hp_excesses, ignore_index=True)
    event_summary = event_summary.sort_values(by="start").reset_index(drop=True)

    event_summary.to_csv(sys.stdout)

    if args.output is not None:
        event_summary.to_csv(args.output, index=False)
        print(f"Output saved to {args.output}")

    if client.influx_client is None:
        await client._influx_client.close()
    else:
        await client.influx_client.close()


def run() -> None:
    asyncio.run(main())
