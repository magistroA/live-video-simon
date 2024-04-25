# Live Video Simon

The project utilizes the YoloV8 and OpenCV video processing libraries to recreate the memory game Simon on a live video feed from a computer webcam. The player will be shown a sequence on the video, and then will be prompted to repeat the sequence by stepping into the correct boxes in the correct order. This project represents my final design project for ECE 1895 Junior Design.

# Play Instruction

### Controls:

``` 
'q' = quit and close video
'r' = restart
```
### Camera Setup:
Run the Camera Test and make sure four colored boxes on the video are all ontop of somewhere the player can stand in each of the four boxes.


### Game Steps:
1. Start the game and wait for the grace period "Welcome to Simon" message to change.
2. The round will begin by indicating a sequence of boxes for the player to memorize accompanied by audible tones, that will continue while the "Wait for Sequence to Complete" message is showing.
3. When the "Repeat the Sequence" message appears, stand in the boxes in the order of the given sequence for the round.
    - To select a box, make sure the players feet and the bottom of the pink player box are in the intended corner. The box will be highlighted and the correct sound when the selection has registered.
    - If a box is selected incorrectly out of sequence, the player will lose.
    - Rounds will also time out and end the game if the player does not select the full sequence in the allotted time.
4. If the player repeats the sequence correctly, a new round will start. Rounds progressively have longer sequences and display the sequences at a faster pace. Play will continue until the player makes a mistake.
5. When the game ends, the players score will display. The player will then be prompted to quit the game or restart.
