import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from music21 import stream, note, midi, converter  # Import converter
import glob

# Step 1: Load and Preprocess MIDI Files
def load_midi_files(midi_folder):
    # Verify the folder exists and list its contents
    if os.path.exists(midi_folder):
        print("Folder exists. Contents:")
        print(os.listdir(midi_folder))  # Print the contents of the MIDI folder
    else:
        print("Folder does not exist.")
        return []

    notes = []
    midi_files = glob.glob(f"{midi_folder}/*.mid")  # Load .mid files only
    print(f"MIDI files found: {midi_files}")  # Check found MIDI files
    
    for file in midi_files:
        print(f"Processing file: {file}")  # Debugging: Show which file is being processed
        try:
            # Use the appropriate way to load the MIDI file
            midi_file = converter.parse(file)  # Use converter to parse the MIDI file
            print(f"Number of parts in {file}: {len(midi_file.parts)}")  # Debugging: Show number of parts
            
            for part in midi_file.parts:  # Iterate through parts instead of tracks
                for element in part.flat.notes:
                    if isinstance(element, note.Note):
                        notes.append(element.pitch.midi)  # Extract MIDI note number
                        print(f"Note extracted: {element.pitch.midi}")  # Print extracted note
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    print(f"Number of notes extracted: {len(notes)}")  # Print the number of notes extracted
    return notes

# Step 2: Prepare Sequences
def prepare_sequences(notes, sequence_length):
    unique_notes = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}
    int_to_note = {number: note for number, note in enumerate(unique_notes)}

    input_sequences = []
    output_notes = []
    
    for i in range(len(notes) - sequence_length):
        input_seq = notes[i:i + sequence_length]
        output_note = notes[i + sequence_length]
        input_sequences.append([note_to_int[note] for note in input_seq])
        output_notes.append(note_to_int[output_note])
        
    n_patterns = len(input_sequences)
    n_vocab = len(unique_notes)

    X = np.reshape(input_sequences, (n_patterns, sequence_length, 1)) / float(n_vocab)
    y = keras.utils.to_categorical(output_notes, num_classes=n_vocab)
    
    return X, y, n_vocab, int_to_note

# Step 3: Build the RNN Model
def create_model(input_shape, n_vocab):
    model = keras.Sequential([
        layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 4: Train the Model
def train_model(model, X, y, epochs=50, batch_size=64):
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Step 5: Generate Music
def generate_music(model, int_to_note, sequence_length, start_sequence, n_notes):
    generated = []
    input_seq = start_sequence
    
    for _ in range(n_notes):
        input_array = np.reshape(input_seq, (1, len(input_seq), 1)) / float(len(int_to_note))
        prediction = model.predict(input_array, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        generated.append(result)
        input_seq.append(index)
        input_seq = input_seq[1:]
    
    return generated

# Step 6: Save Generated Music to MIDI
def save_to_midi(generated_sequence, output_file):
    midi_stream = stream.Stream()
    for note_number in generated_sequence:  # Rename the loop variable for clarity
        midi_note = note.Note(note_number)  # Use note.Note correctly
        midi_stream.append(midi_note)
    midi_stream.write('midi', fp=output_file)
# Main Function
if __name__ == "__main__":
    midi_folder = "/Users/shivaym/Desktop/shivaay/Experience/VS CODE PROJECTS/music systm/midi_files"  # Update the path

    n_notes_to_generate = 200
    sequence_length = 100  # Define your sequence length

    # Load and preprocess MIDI files
    notes = load_midi_files(midi_folder)
    if not notes:
        print("No notes extracted. Please check your MIDI files.")
    else:
        X, y, n_vocab, int_to_note = prepare_sequences(notes, sequence_length)

        # Create and train the model
        model = create_model((X.shape[1], X.shape[2]), n_vocab)
        train_model(model, X, y)

        # Generate music
        start_sequence = notes[:sequence_length]  # Use the first few notes as a seed
        generated_notes = generate_music(model, int_to_note, sequence_length, [note for note in start_sequence], n_notes_to_generate)

        # Save generated music to MIDI file
        save_to_midi(generated_notes, "generated_music.mid")
        print("Generated music saved to 'generated_music.mid'.")
