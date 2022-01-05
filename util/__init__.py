import datetime
import strict_rfc3339


def epoch_to_str(dt, smallest):
    if smallest == 'ms':
        dt = dt / 1000
    if smallest == 'us':
        dt = dt / 1000 / 1000
    return strict_rfc3339.timestamp_to_rfc3339_utcoffset(dt)


# todo: optimize round_time (redo without datetime conversion)
def round_time(dt, date_delta=datetime.timedelta(minutes=1)):
    if dt == 0:
        return 0
    dt = datetime.datetime.fromtimestamp(dt / 1000)
    round_to = date_delta.total_seconds()
    seconds = (dt - dt.min).seconds

    if seconds % round_to == 0 and dt.microsecond == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        rounding = seconds // round_to * round_to

    return int((dt + datetime.timedelta(0, rounding - seconds, - dt.microsecond)).timestamp())
