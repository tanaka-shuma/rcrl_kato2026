from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)
core.warmup()

ok = True
total_spikes = 0

for k in range(20):
    t_start = k * 50.0
    t_end = (k + 1) * 50.0

    core.advance(50.0)

    ids = core.get_step_spike_ids()
    times = core.get_step_spike_times()

    total_spikes += len(ids)

    if len(times) > 0:
        if not ((times >= t_start).all() and (times < t_end).all()):
            print(
                "time error:",
                k + 1,
                times.min(),
                times.max()
            )
            ok = False

    if len(ids) != len(times):
        print("length error:", k + 1)
        ok = False

    print(
        f"step {k+1:2d}: "
        f"elapsed={core.get_elapsed_ms():6.1f} ms, "
        f"spikes={len(ids):4d}, "
        f"time range={times[:1]} ... {times[-1:]}"
    )

print("elapsed:", core.get_elapsed_ms())
print("total spikes:", total_spikes)
print("check:", ok)

core.advance(50.0)

print("after boundary")
print("elapsed:", core.get_elapsed_ms())
print("times:", core.get_step_spike_times()[:20])