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

import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient

from lsst.ts.m1m3.utils import HPForces


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
    parser = argparse.ArgumentParser(description="Calculate HP excesses for a given definition of excess")
    parser.add_argument("--efd", default="usdf_efd", help="EFD name. Defaults to usdf_efd.")
    parser.add_argument(
        "--t1",
        type=str,
        default="2025-12-30T00:00:00Z",
        help="Start time for excess check in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "--t2",
        type=str,
        default="2025-12-30T00:59:59Z",
        help="End time for excess check in a valid format: 'YYYY-MM-DDTHH:MM:SSZ'",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=1,
        help="Reference period (in seconds) in which an 'excess' is defined",
    )
    parser.add_argument(
        "--delta_f_threshold",
        type=float,
        default=200,
        help="Minimum increment of force (in N) defining an excess (in delta_t)",
    )
    parser.add_argument(
        "--time_gap_threshold",
        type=str,
        default="1s",
        help="Minimum time between events to consider a grouped excess. 1s is a good value.",
    )
    parser.add_argument(
        "--hp_id",
        type=int,
        default=2,
        help="Hardpoint identifier [1-6]",
    )
    args = parser.parse_args()
    client = EfdClient(args.efd)

    hp_index = args.hp_id - 1
    df = await get_data(client, f"measuredForce{hp_index}", Time(args.t1), Time(args.t2))
    sampling_freq = 20  # Hz
    time_gap_threshold = TimeDelta(args.time_gap_threshold)
    hpf = HPForces(df, sampling_freq, time_gap_threshold)
    event_summary = hpf.calculate_excesses(args.hp_id, args.delta_t, args.delta_f_threshold)
    output_filename = f"event_summary_HP{args.hp_id}_{args.t1}_{args.t2}.txt"
    event_summary.to_csv(output_filename, sep="\t", index=False)
    print(f"Output saved to {output_filename}")

    if client.influx_client is None:
        await client._influx_client.close()
    else:
        await client.influx_client.close()


def run() -> None:
    asyncio.run(main())
