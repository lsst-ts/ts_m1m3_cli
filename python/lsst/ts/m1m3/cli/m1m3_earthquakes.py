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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient

from lsst.ts.m1m3.utils import DurationTime


@dataclass
class Earthquake:
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
    zero_tolerance: float = 10.0,
    abs_sum_threshold: float = 600.0,
    efd_instance: str = "usdf_efd",
    topic_name: str = "lsst.sal.MTM1M3.hardpointActuatorData",
) -> None:
    """
    Queries MTM1M3 EFD telemetry data, calculates rolling sums and absolute
    sums, and identifies potential earthquake signatures where forces oscillate
    around zero.
    """
    # 1. Initialize the EFD Client
    client = EfdClient(efd_instance)

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

                def count_sign_changes(window: np.array) -> int:
                    signs = np.sign(window)
                    signs = signs[signs != 0]
                    return (signs[:-1] != signs[1:]).sum()

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
            import traceback

            traceback.print_exc()

            logging.error(f"Error querying chunk {chunk_start} to {chunk_end}: {e}")

        # Move forward to the next chunk
        chunk_start = chunk_end

    # Filter and output detected events

    last_event = dict(zip(target_cols, [None] * len(target_cols)))

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
            else:
                logging.debug(
                    f"Not printing: {earthquake.component}: {earthquake.timestamp} {earthquake.plain_sum} "
                    f"{earthquake.abs_sum} {earthquake.sign_changes}"
                )

            last_event[earthquake.component] = earthquake.timestamp
    else:
        print("\nNo earthquake signatures matched the criteria in this time range.")


def parse_arguments() -> argparse.Namespace:
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

    asyncio.run(
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
