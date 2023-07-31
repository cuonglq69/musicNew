import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional, LSTM
from keras.optimizers import Adam
from collections import Counter
import random

def collect_notes():
    notes = []

    files = glob.glob("midi_file_bethoven/*.mid")

    random_files = random.sample(files, 3)  # Randomly select 3 files from the folder

    for file in random_files:
        midi = converter.parse(file)
        print("Processing file:", file)

        picked_notes = None
        # If a file includes instrument parts
        try:
            track = instrument.partitionByInstrument(midi)
            picked_notes = track.parts[0].recurse()
        # If a file does not have instrument parts and instead has a flat structure containing notes directly
        except:
            picked_notes = midi.flat.notes

        for element in picked_notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def convert_to_midi(music):
    melody = []
    offset = 0
    for item in music:
        if "." in item or item.isdigit():
            chord_has_notes = item.split(".")
            notes = []
            for note_item in chord_has_notes:
                note_item_to_int = int(note_item)
                note_set = note.Note(note_item_to_int)
                notes.append(note_set)
            chord_set = chord.Chord(notes)
            chord_set.offset = offset
            melody.append(chord_set)
        else:
            note_set = note.Note(item)
            note_set.offset = offset
            melody.append(note_set)
        offset += 1
    melody_midi = stream.Stream(melody)
    return melody_midi

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Dropout(0.1))
    model.add(Dense(output_shape, activation='softmax'))
    opt = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()
    return model


def train_model(model, X_train, y_train, epochs=350, batch_size=64):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

def generate_song(model, seed_notes, unique_notes, note_to_int, int_to_note, length=100):
    generated_notes = seed_notes.copy()
    input_sequence = [note_to_int[note] for note in seed_notes]

    for _ in range(length):
        input_sequence_array = np.reshape(input_sequence, (1, len(input_sequence), 1))
        input_sequence_array = input_sequence_array / float(len(unique_notes))

        prediction = model.predict(input_sequence_array, verbose=0)[0]
        index = np.argmax(prediction)
        result_note = int_to_note[index]
        generated_notes.append(result_note)
        input_sequence.append(index)
        input_sequence = input_sequence[1:]

    generated_melody = convert_to_midi(generated_notes)
    return generated_melody


def run_training():
    notes = collect_notes()
    print("Total notes in all music files:", len(notes))

    count_freq = Counter(notes)
    infreq_notes = [note for note, count in count_freq.items() if count < 100]
    notes = [note for note in notes if note not in infreq_notes]

    unique_notes = sorted(list(set(notes)))
    num_notes = len(notes)
    num_unique_notes = len(unique_notes)

    note_to_int = dict((note, index) for index, note in enumerate(unique_notes))
    int_to_note = dict((index, note) for index, note in enumerate(unique_notes))

    input_sequences = []
    output_notes = []
    seq_length = 40

    for i in range(0, num_notes - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        input_sequences.append([note_to_int[note] for note in seq_in])
        output_notes.append(note_to_int[seq_out])

    num_sequences = len(input_sequences)

    if num_sequences == 0:
        raise ValueError("No sequences available for training.")

    input_sequences_array = np.reshape(input_sequences, (num_sequences, seq_length, 1))
    input_sequences_array = input_sequences_array / float(num_unique_notes)

    output_notes_array = tensorflow.keras.utils.to_categorical(output_notes)

    X_train, X_test, y_train, y_test = train_test_split(input_sequences_array, output_notes_array, test_size=0.2,
                                                        random_state=42)

    model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    train_model(model, X_train, y_train)

    model_filename = "model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    return model, unique_notes, note_to_int, int_to_note


def load_and_generate(model, unique_notes, note_to_int, int_to_note, seed_notes, length=100):
    # with open(model_filename, 'rb') as file:
    #     model = pickle.load(file)
    generated_melody = generate_song(model, seed_notes, unique_notes, note_to_int, int_to_note, length)
    return generated_melody
