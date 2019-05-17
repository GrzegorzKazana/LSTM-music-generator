import sys
import numpy as np
import random
from math import ceil
from mido import MidiFile, MidiTrack, Message, MetaMessage
if __name__ == '__main__':
    from common import MSECS_PER_FRAME, NUM_NOTES, NUM_VELOCITY, DEFAULT_BPM, compose, debug, read_numpy_midi
else:
    from .common import MSECS_PER_FRAME, NUM_NOTES, NUM_VELOCITY, DEFAULT_BPM, compose, debug, read_numpy_midi


def to_delta_time(messages):
    """
    takes messages with absolute timestamp, and returns list of 
    messages with delta time
    """
    messages_detlta_time = []
    now = 0
    for msg in messages:
        delta = int(msg.time - now)
        new_message = msg.copy(time=delta)
        messages_detlta_time.append(new_message)
        now = msg.time
    return messages_detlta_time


def total_decode(encoded_numpy):
    """
    takes in
    [[one_hot_encoded_note, velocities], ...] (n_of_frames x (one_hot_encoded_note + velocities))
    and transforms to
    [Messages with absolute time stamp]
    note_offs are encoded as note_ons with vel=0
    """
    messages = []
    # handle first frame
    notes_on = np.argwhere(encoded_numpy[0, :NUM_NOTES] == 1).flatten()
    for note in notes_on:
        print(note)
        print(int(encoded_numpy[0, NUM_NOTES + note] * 128))
        messages.append(Message('note_on', note=note,
                                velocity=int(encoded_numpy[0, NUM_NOTES + note] * 128), time=0))

    for i in range(1, len(encoded_numpy)):
        prev_frame = encoded_numpy[i-1]
        curr_frame = encoded_numpy[i]
        diff = curr_frame - prev_frame
        diff_notes = diff[:NUM_NOTES]
        notes_on = np.argwhere(diff_notes == 1).flatten()
        notes_off = np.argwhere(diff_notes == -1).flatten()

        for note in notes_on:
            time = i * MSECS_PER_FRAME
            messages.append(Message(
                'note_on', note=note, velocity=int(curr_frame[NUM_NOTES + note] * 128), time=time))
        for note in notes_off:
            time = i * MSECS_PER_FRAME
            messages.append(Message('note_on', note=note,
                                    velocity=0, time=time))
    return messages


def messages_to_midifile(messages):
    """
    creates MidiFile with given messages
    """
    outfile = MidiFile()
    track = MidiTrack()
    outfile.tracks.append(track)
    for msg in messages:
        track.append(msg)
    return outfile


def pipe_reverse(raw_numpy):
    return compose(
        total_decode,
        to_delta_time,
        messages_to_midifile
    )(raw_numpy)


if __name__ == '__main__':
    # parsing arguments
    arguments = sys.argv[1:]
    if len(arguments) < 1 or (
        '.csv' not in arguments[0] and
        '.npy' not in arguments[0] and
        '.npz' not in arguments[0]
    ):
        raise Exception('Please specify valid file path')
    input_path = arguments[0]

    if len(arguments) < 2 or ('.mid' not in arguments[1] and '.midi' not in arguments[1]):
        raise Exception('Invalid output path')
    output_path = arguments[1]

    # reading file
    raw_numpy = read_numpy_midi(input_path)

    # processing and save
    mid = pipe_reverse(raw_numpy)
    mid.save(output_path)
