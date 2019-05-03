MSECS_PER_FRAME = 10
NUM_NOTES = 128
NUM_VELOCITY = 128
DEFAULT_BPM = 120


def compose(*functions):
    """
    composes functions left to right
    """
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg
    return inner


def debug(messages):
    if 'numpy' in str(type(messages)):
        print(messages.shape)
        print(messages)
    else:
        for msg in messages:
            print(msg)
        print(len(messages))
    return messages
