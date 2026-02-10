import pandas as pd
import argparse
import asyncio
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
        default="2025-12-30T23:59:59Z",
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
        default=0,
        help="Hardpoint identifier [0-5]",
    )
    args = parser.parse_args()
    client = EfdClient(args.efd)
    df = await get_data(client, f"measuredForce{args.hp_id}", Time(args.t1), Time(args.t2))
    sampling_freq = 20  # Hz
    time_gap_threshold = TimeDelta(args.time_gap_threshold)
    hpf = HPForces(df, sampling_freq, time_gap_threshold)
    event_summary = hpf.calculate_excesses(args.hp_id, args.delta_t, args.delta_f_threshold)
    event_summary.to_csv(f"event_summary_HP{args.hp_id}_{args.t1}_{args.t2}.txt", sep="\t", index=False)


if __name__ == "__main__":
    asyncio.run(main())
