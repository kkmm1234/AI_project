package ie.atu.sw;

import java.util.LinkedList;

//created a deterministic rules based bot to optimise data collection
public class Bot {
	//method to decide move based on the game
    public static int decideMove(LinkedList<byte[]> model, int playerRow, int lastMove) {
        //get the next column from the user (player column 15)
        int nextColumnIndex = (15 + 1) % model.size();
        //get the data from this colum
        byte[] nextColumn = model.get(nextColumnIndex);
        
        //check if we are going to hit a wall
        if (playerRow < nextColumn.length && nextColumn[playerRow] != 0) {
            //look for open space
            int up = playerRow - 1;
            int down = playerRow + 1;
            
            //loop for top of next column and while we havent hit it
            while (up >= 0 && nextColumn[up] != 0) {
                up--; // move up
            }
            
            //loop for bottom of next column while we havent hit it
            while (down < nextColumn.length && nextColumn[down] != 0) {
                down++; //move down
            }
            
            //if both directions are clear choose the closest one
            if (up >= 0 && down < nextColumn.length) {
                return (playerRow - up < down - playerRow) ? -1 : 1;
            } 
            
            //if only up has empty space 
            else if (up >= 0) {
                return -1;//move up
            } 
            //if only down has empty space
            else if (down < nextColumn.length) {
                return 1;//move down
            }
            //if we have no clear path (just for safety
            else {
                return 0;//stay in place
            }
        }
        
        //if there is no wall ahead fly straight
        return 0;
    }
}