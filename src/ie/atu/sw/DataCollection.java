package ie.atu.sw;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;

public class DataCollection {
    //extract set features from game
    public static double[] extractFeatures(LinkedList<byte[]> model, int playerRow, int lastMove) {
        //set how far to look forward for collection (5 columns)
    	int horizon = 5;
    	//we have 5 features per column so we multiply but 5 the add 2 for playerRow and last move
        double[] features = new double[horizon * 5 + 2];

        //index to keep track of position in feature array
        int featureIndex = 0;

        //loop through each column in horizon
        for (int i = 0; i < horizon; i++) {
        	//calculate the index of the column (15 is player column)
            int colIndex = (15 + i + 1) % model.size();
            //get the column from the game
            byte[] column = model.get(colIndex);
            
            //top boundary
            int top = 0;
            //find the top boundary by looping through looking for first 0
            while (top < column.length && column[top] != 0) top++;

            //bottom boundary
            int bottom = column.length - 1;
            //look for last 0 this is bottom boundary
            while (bottom >= 0 && column[bottom] != 0) bottom--;

            //calculate the center just add the top and bottom to get safe area divide by 2
            int center = (top + bottom) / 2;
            //width of the cave
            int width = bottom - top;
            //distance player is the the center of cave
            int distanceToCenter = playerRow - center;

            //add these to the features array
            features[featureIndex++] = top;
            features[featureIndex++] = bottom;
            features[featureIndex++] = center;
            features[featureIndex++] = width;
            features[featureIndex++] = distanceToCenter;
        }
        
        //add the player row and last move to features also
        features[featureIndex++] = playerRow;
        features[featureIndex++] = lastMove;

        return features;
    }

    //method takes features captured in collection writes them to file
    public static void saveToCSV(double[] features, int label) {
    	//ensure resource folder exist
        File resourcesDir = new File("resources");
        if (!resourcesDir.exists()) {
            resourcesDir.mkdir();
        }
        
    	//find file to write to 
        File trainingFile = new File(resourcesDir, "training_data.csv");
        
        try {

            //ensure file exists
            boolean fileExists = trainingFile.exists();

            //create a file write with append set to true so more data can be collected
            FileWriter fw = new FileWriter(trainingFile, true);

            //if file does not already exist write a header
            if (!fileExists) {
                fw.write("top1,bottom1,center1,width1,dist1,"
                       + "top2,bottom2,center2,width2,dist2,"
                       + "top3,bottom3,center3,width3,dist3,"
                       + "top4,bottom4,center4,width4,dist4,"
                       + "top5,bottom5,center5,width5,dist5,"
                       + "playerRow,lastMove,label\n");
            }

            //make a string builder to build features into a string
            StringBuilder sb = new StringBuilder();
            //loop through all the features
            for (double value : features) {
            	//add comma to separate features 
                sb.append(value).append(",");
            }

            //add the label to end of the feature string
            sb.append(label).append("\n");
            //write the data to the file
            fw.write(sb.toString());
            fw.close();
        } catch (IOException e) {
            System.out.println("error saving data: " + e.getMessage());
        }
    }

}