import numpy as np
from math import ceil
from mido import MidiFile

SELECTED_CHANNEL = 1
MSECS_PER_FRAME = 10


def compose(*functions):
    """
    composes functions left to right
    """
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg
    return inner


def to_absolute_time(messages):
    """
    takes in list of messages with delta-time timestamp,
    and returns list of messages with absolute timestamp
    """
    accumulated_time = 0
    messages_abs_time = []
    for msg in messages:
        accumulated_time += msg.time
        new_msg = msg.copy(time=accumulated_time)
        messages_abs_time.append(new_msg)
    return messages_abs_time


def filter_channel(channel_n, messages):
    """
    filters list of messages to only selected channel
    """
    return filter(lambda msg: msg.channel == channel_n, messages)


def filter_meta(messages):
    """
    filters list of messages to note_on and note_off events
    """
    return filter(lambda msg: msg.type == 'note_on' or msg.type == 'note_off', messages)


def note_off_to_zero_vel(messages):
    """
    changes note_off events to note_on events with 0 velocity
    """
    return map(lambda msg: msg.copy(type='note_on', velocity=0) if msg.type == 'note_off' else msg, messages)


def secs_to_msecs(messages):
    """
    transforms messages' timestamp from secs to msecs
    """
    return [msg.copy(time=1000 * msg.time) for msg in messages]


def encode_message(msg):
    """
    encodes mido message object to list
    """
    return [msg.note, msg.velocity, msg.time]


def to_raw_numpy(messages):
    """
    encodes all messages,
    and converts it to numpy array
    """
    return np.array([encode_message(msg) for msg in messages], dtype=np.int32)


def msecs_to_frames(raw_numpy):
    """
    processes time data from msecs to frame time,
    assumes time stamp is in last column
    """
    raw_numpy[:, -1] //= MSECS_PER_FRAME
    return raw_numpy


def snip_track(raw_numpy):
    """
    if first note starts not in first frame,
    every note is shifted
    """
    offset = raw_numpy[0, -1]
    raw_numpy[:, -1] -= offset
    return raw_numpy


def total_encode(raw_numpy):
    """
    transforms 
    [[note, velocity, time], ...]   (n_of_notes x 3)
    to
    [[one_hot_encoded_note, velocity], ...] (n_of_frames x (one_hot_encoded_note + velocity))
    """
    NUM_NOTES = 128
    NUM_VELOCITY = 1
    n_of_frames = raw_numpy[-1, -1]
    encoded = np.zeros((n_of_frames, NUM_NOTES + NUM_VELOCITY))
    for note, velocity, time in raw_numpy:
        if velocity == 0:
            encoded[time:, note] = 0
        else:
            encoded[time:, note] = 1
            encoded[time:, -1] = velocity / 128

    return encoded


def pipe(track):
    return compose(
        note_off_to_zero_vel,
        secs_to_msecs,
        to_absolute_time,
        filter_meta,
        lambda x: filter_channel(SELECTED_CHANNEL, x),
        to_raw_numpy,
        msecs_to_frames,
        total_encode,
    )(track)


if __name__ == '__main__':
    mid = MidiFile(
        'D:\Programowanie\Datasets\MIDI\pop_midi_dataset_ismir\midis\Guitar_midkar.com_MIDIRip\jazz\\take_5_jh.mid')
    all_messages = [msg for msg in mid]
    res = pipe(all_messages)
    print(res)
