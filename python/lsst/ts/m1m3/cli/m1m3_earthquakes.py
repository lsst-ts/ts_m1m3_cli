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

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass
from traceback import print_exc

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient
from numba import jit

from lsst.ts.m1m3.utils import DurationTime


@dataclass
class Earthquake:
    """Earthquake record. Stores unfiltered events, so sign_changes is included
    as a counter - it counts number of sign changes, occuring later in the
    filtering.

    Attributes
    ----------
    timestamp : `Time`
        Time of the event.
    component : `str`
        Name of component - shall be either f[xyz] for forces, or m[xyz] for
        momements.
    plain_sum : `float`
        Plain (signed) sum of force values in test window.
    abs_sum : `float`
        Sum of absolute values in test window.
    sign_changes : `float`
        Sign changes counter.
    """

    timestamp: Time
    component: str
    plain_sum: float
    abs_sum: float
    sign_changes: int


async def detect_earthquake_signals(
    start_time: Time,
    end_time: Time,
    chunk_timedelta: TimeDelta = TimeDelta(3600, format="sec"),
    n_measurements: int = 10,
    min_sign_changes: int = 3,
    zero_tolerance: float = 1000.0,
    abs_sum_threshold: float = 6000.0,
    efd_instance: str = "usdf_efd",
) -> pd.DataFrame:
    """
    Queries MTM1M3 EFD telemetry data, calculates rolling sums and absolute
    sums, and identifies potential earthquake signatures where forces oscillate
    around zero.

    Parameters
    ----------
    start_time : `Time`
        Interval search start time.
    end_time : `Time`
        Interval search end time.
    chunk_timedelta : `TimeDelta`
        Duration of chunk for EFD queries (EFD cannot return huge amount of
        data). Defaults to 1 hour.
    n_measurements : `int`
        Window width in measurements. As the M1M3 runs on 50 Hz, a measurement
        is available every 20 milliseconds. Defaults to 10.
    min_sign_changes : `int`
        Minimal number of sign changes for filtering. If the wave produced
        isn't fast enough, it will be filtered out. Defaults to 3.
    zero_tolerance : `float`
        Tolerance for zero crossing in N. Defaults to 1000.0. The higher the
        value, the higher would be number of event - including false triggers.
        Suggested value is 100.0 * n_measurements.
    abs_sum_threshold : `float`
        Threshold for absolute sum in N. Defauts to 6000.0. The higher the
        value, the lower number of events. Suggested value is 600.0 *
        n_measurements.
    efd_instance : `str`
        EFD instance to query. Defaults to usdf_efd.

    Returns
    -------
    data : `pd.DataFrame`
        Dataframe of filtered Earthquake events. Earthquake attributes form
        columns labels.
    """
    # 1. Initialize the EFD Client
    client = EfdClient(efd_instance)

    topic_name = "lsst.sal.MTM1M3.hardpointActuatorData"

    logging.info(f"Connecting to EFD ({efd_instance})...")
    print(f"Querying topic '{topic_name}' from {start_time} to {end_time} (UTC)...")
    logging.debug(f"Chunk size limit: {chunk_timedelta} seconds.\n")

    target_cols = ["fx", "fy", "fz", "mx", "my", "mz"]

    alerts_triggered: list[Earthquake] = []
    chunk_start = start_time
    previous_df: None | pd.DataFrame = None

    # Loop through time range in chunks
    while chunk_start < end_time:
        chunk_end = min(chunk_start + chunk_timedelta, end_time)

        logging.debug(f"Fetching chunk: {chunk_start} to {chunk_end}...")

        try:
            # Query the current chunk
            df = await client.select_time_series(
                topic_name, ["timestamp"] + target_cols, chunk_start, chunk_end
            )

            if df.empty:
                logging.warning(f"No data in chunk {chunk_start} to {chunk_end}, skipping.")
                chunk_start = chunk_end
                previous_df = None
                continue

            logging.debug(
                f"Retrieved {len(df)} rows. Processing running sums for window n = {n_measurements}..."
            )

            if previous_df is not None:
                gap = df.index[0] - previous_df.index[-1]
                if gap < pd.Timedelta(milliseconds=40):
                    df = pd.concat([previous_df, df])
                    logging.debug(
                        f"Continuous stream detected (gap: {gap.total_seconds() * 1000:.1f} ms). "
                        f"Carried over {len(previous_df)} rows."
                    )
                else:
                    logging.warning(
                        f"Data gap too large (gap: {gap.total_seconds() * 1000:.1f} ms), "
                        "starting fresh rolling window."
                    )

            # Evaluate rolling windows for each column
            for col in target_cols:
                # Calculate plain running sum and absolute running sum on data

                plain_window = df[col].rolling(window=n_measurements)

                plain_sum = plain_window.sum()
                abs_sum = df[col].abs().rolling(window=n_measurements).sum()

                # jit speeds-up processing by ~10%, so worth the hassle
                @jit
                def count_sign_changes(window: np.array) -> int:
                    signs = np.sign(window)
                    signs = signs[signs != 0]
                    return (signs[:-1] != signs[1:]).sum()

                # raw to speed-up computation - surprisingly, that takes more
                # time than estimated
                sign_changes = plain_window.apply(count_sign_changes, raw=True)

                # Condition: Plain sum is close to 0 (within `zero_tolerance`)
                # AND absolute sum exceeds `abs_sum_threshold`
                condition = (
                    (plain_sum.abs() <= zero_tolerance)
                    & (abs_sum > abs_sum_threshold)
                    & (sign_changes >= min_sign_changes)
                )

                matched_rows = df[condition]
                for timestamp in matched_rows.index:
                    alerts_triggered.append(
                        Earthquake(
                            timestamp, col, plain_sum[timestamp], abs_sum[timestamp], sign_changes[timestamp]
                        )
                    )

            previous_df = df[-n_measurements + 1 :].copy()

        except Exception as e:
            print_exc()

            logging.error(f"Error querying chunk {chunk_start} to {chunk_end}: {e}.")

        # Move forward to the next chunk
        chunk_start = chunk_end

    # Filter and output detected events

    last_event = dict(zip(target_cols, [None] * len(target_cols)))

    filtered_events: list[Earthquake] = []

    if len(alerts_triggered) > 0:
        print("\n--- Detection Results ---")
        for earthquake in sorted(alerts_triggered, key=lambda earthquake: earthquake.timestamp):
            # find closest in-time non-triggered axis

            times: list[Time] = [
                last_event[k] for k in last_event if last_event[k] is not None and k != earthquake.component
            ]

            if len(times) > 0 and (earthquake.timestamp - max(times)) < TimeDelta(
                0.021 * n_measurements, format="sec"
            ):
                print(
                    f"{earthquake.timestamp} earthquake ("
                    f"component: {earthquake.component}, "
                    f"plain sum: {earthquake.plain_sum:.1f}, "
                    f"abs_sum: {earthquake.abs_sum:.1f}, "
                    f"sign_changes: {earthquake.sign_changes})"
                )
                filtered_events.append(earthquake)
            else:
                logging.debug(
                    f"Filtered out: {earthquake.component}: {earthquake.timestamp} {earthquake.plain_sum} "
                    f"{earthquake.abs_sum} {earthquake.sign_changes}"
                )

            last_event[earthquake.component] = earthquake.timestamp
    else:
        print("\nNo earthquake signatures matched the criteria in this time range.")

    return pd.DataFrame([asdict(x) for x in filtered_events])


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    args : `argparse.Namespace`
        Parsed arguments.
    """
    now = Time.now()

    parser = argparse.ArgumentParser(description="Look for possible seismic events.")

    parser.add_argument(
        "start_time",
        type=DurationTime(now),
        default=DurationTime(now - TimeDelta(180, format="sec")),
        nargs="?",
        help="Start time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )
    parser.add_argument(
        "end_time",
        type=DurationTime(now),
        default=DurationTime(now),
        nargs="?",
        help="End time in a valid format: 'YYYY-MM-DD HH:MM:SSZ'",
    )

    parser.add_argument(
        "--efd",
        default="usdf_efd",
        help="EFD name. Defaults to usdf_efd",
    )
    parser.add_argument(
        "-d",
        default=False,
        action="store_true",
        help="Print debug messages",
    )

    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Rolling sample size. Defaults to 10.",
    )
    parser.add_argument("--sign-changes", type=int, default=3, help="Minimal sign changes. Default to 3.")
    parser.add_argument(
        "-a",
        type=float,
        default=600.0,
        help="Absolute tolerance. Defaults to 600.0",
    )
    parser.add_argument(
        "-s",
        type=float,
        default=100.0,
        help="Minimum sum value to trigger earthquake. Defaults to 100.0.",
    )
    parser.add_argument(
        "-o",
        type=argparse.FileType("w"),
        dest="output",
        help="Save filtered events to this file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=3600.0,
        help="Query chunk duration in seconds. Defaults to 3600 (1 hour).",
    )

    return parser.parse_args()


def run() -> None:
    args = parse_arguments()

    start_t, end_t = DurationTime.pair(args.start_time, args.end_time)

    level = logging.DEBUG if args.d else logging.INFO
    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    filtered_df = asyncio.run(
        detect_earthquake_signals(
            start_time=start_t,
            end_time=end_t,
            chunk_timedelta=TimeDelta(args.chunk_size, format="sec"),
            n_measurements=args.n,
            min_sign_changes=args.sign_changes,
            zero_tolerance=args.s * args.n,
            abs_sum_threshold=args.a * args.n,
        )
    )

    if args.output is not None:
        file_name = args.output.name if hasattr(args.output, "name") else args.output
        filtered_df.to_csv(args.output, index=False)
        logging.info(f"Saved {len(filtered_df.index)} records to {file_name}.")
