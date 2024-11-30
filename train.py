import tensorflow as tf
import numpy as np
import chess
import chess.pgn
import random
import os
import sys
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import random
import chess
import random
import numpy as np
import tensorflow as tf

def board_to_tensor(board):
    """Convert chess board to a numerical tensor representation."""
    # 8x8x13 tensor representation
    # 12 piece types (6 white, 6 black), 1 for turn, 1 for castling rights
    board_tensor = np.zeros((8, 8, 13), dtype=np.float32)
    
    # Piece mapping
    piece_indices = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Populate board tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            # Determine color and piece type index
            color_offset = 6 if piece.color == chess.BLACK else 0
            piece_index = piece_indices[piece.piece_type] + color_offset
            
            board_tensor[row, col, piece_index] = 1
    
    # Add turn information
    board_tensor[:, :, 12] = 1 if board.turn == chess.WHITE else 0
    
    return board_tensor

def move_to_tensor(move):
    """Convert chess move to a numerical tensor representation."""
    # One-hot encoding of move
    move_tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    
    # Get source and destination squares
    from_square = move.from_square
    to_square = move.to_square
    
    # Convert squares to row and column
    from_row = chess.square_rank(from_square)
    from_col = chess.square_file(from_square)
    to_row = chess.square_rank(to_square)
    to_col = chess.square_file(to_square)
    
    # Mark the move
    move_tensor[from_row, from_col, to_row, to_col] = 1
    
    return move_tensor

def generate_random_data(num_samples, max_moves_per_game=50):
    """Generate random chess game data."""
    positions = []
    moves = []
    
    for _ in range(num_samples):
        # Reset board
        board = chess.Board()
        
        # Play random moves
        for _ in range(max_moves_per_game):
            # Get legal moves
            legal_moves = list(board.legal_moves)
            
            # Check if legal moves exist
            if not legal_moves:
                break
            
            # Choose a random legal move
            chosen_move = random.choice(legal_moves)
            
            # Convert board and move to tensors
            board_tensor = board_to_tensor(board)
            move_tensor = move_to_tensor(chosen_move)
            
            # Store data
            positions.append(board_tensor)
            moves.append(move_tensor)
            
            # Make the move
            board.push(chosen_move)
            
            # Optional: Break if game ends
            if board.is_game_over():
                break
    
    return np.array(positions), np.array(moves)

def create_chess_model():
    """Create a neural network model for chess move prediction."""
    input_shape = (8, 8, 13)
    move_shape = (8, 8, 8, 8)
    
    # Input layer for board state
    board_input = tf.keras.layers.Input(shape=input_shape)
    
    # Convolutional layers
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(board_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer for move prediction
    move_output = tf.keras.layers.Dense(move_shape[0]*move_shape[1]*move_shape[2]*move_shape[3], 
                                        activation='sigmoid')(x)
    move_output = tf.keras.layers.Reshape(move_shape)(move_output)
    
    # Create model
    model = tf.keras.Model(inputs=board_input, outputs=move_output)
    
    # Compile model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Generate training data
    print("Generating training data...")
    positions, moves = generate_random_data(3000)
    
    # Create and train model
    print("Creating chess model...")
    model = create_chess_model()
    
    print("Training model...")
    model.fit(positions, moves, 
              epochs=10, 
              batch_size=32, 
              validation_split=0.2)
    
    # Save model
    model.save('chess_move_predictor.h5')
    print("Model training complete!")

if __name__ == "__main__":
    main()

# Convert board to a numeric state
def board_to_state(board):
    """Convert a chess.Board object to a numeric array."""
    state = np.zeros((8, 8, 12))  # 8x8 board with 12 possible piece types
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        state[row, col, piece_map[str(piece)]] = 1
    return state

# Generate random data (for demonstration purposes)
def generate_random_data(num_samples=3000):
    """Generate random chess positions and corresponding moves."""
    positions = []
    moves = []
    for _ in range(num_samples):
        board = chess.Board()
        for _ in range(random.randint(0, 20)):  # Play a few random moves
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        
        positions.append(board_to_state(board))
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        moves.append(chosen_move.uci())
    return positions, moves

# Prepare data
positions, moves = generate_random_data(3000)

# Convert UCI moves to a one-hot encoding
def moves_to_one_hot(moves):
    uci_dict = {uci: i for i, uci in enumerate(sorted(set(moves)))}
    move_hot = np.zeros((len(moves), len(uci_dict)))
    for i, move in enumerate(moves):
        move_hot[i, uci_dict[move]] = 1
    return move_hot, uci_dict

moves_hot, uci_dict = moves_to_one_hot(moves)
positions = np.array(positions)

# The board_to_state() function creates a state with shape (8, 8, 12)
inputs = tf.keras.Input(shape=(8, 8, 12))

# Or if you want to flatten the input
inputs = tf.keras.Input(shape=(8 * 8 * 12,))

# Full model example:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(8, 8, 12)),  # Flatten the input
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(uci_dict), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(positions, moves_hot, epochs=1000, batch_size=32)

# Save the model
model.save("chess_bot.h5")

# Usage example
def predict_move(board, model, uci_dict):
    state = np.expand_dims(board_to_state(board), axis=0)
    prediction = model.predict(state)
    predicted_move_index = np.argmax(prediction)
    inv_uci_dict = {v: k for k, v in uci_dict.items()}
    return inv_uci_dict[predicted_move_index]

# Test the model
test_board = chess.Board()
predicted_move = predict_move(test_board, model, uci_dict)
print("Predicted move:", predicted_move)
