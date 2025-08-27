package ie.atu.sw;

import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class NerualNetwork {
    private BasicNetwork network;
    
    public void createNetwork() {
        //creating a network 27 inputs 3 hidden layers (neurons: 32, 24, 16) and 3 outputs
        network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 27)); //input layer 27 inputs for each feature
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 32)); //hidden layer 1
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 24)); //hidden layer 2
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 16)); //hidden layer 3
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, 3)); //output
        network.getStructure().finalizeStructure();
        network.reset();
    }
    
    //load training data and train model
    @SuppressWarnings("resource")
	public void loadAndTrain() {
        try {
        	//save number of different inputs for displaying
        	int up = 0, stay = 0, down = 0;
        	//ensure file is not empty
        	boolean fileEmpty = false;
        	
        	//Lists to store input features for each movement for balancing
        	List<double[]> inputUp = new ArrayList<>();
            List<double[]> inputStay = new ArrayList<>();
            List<double[]> inputDown = new ArrayList<>();
            List<double[]> outputUp = new ArrayList<>();
            List<double[]> outputStay = new ArrayList<>();
            List<double[]> outputDown = new ArrayList<>();
            
            //ensure resources folder exists
            File resourcesDir = new File("resources");
            if (!resourcesDir.exists()) {
                resourcesDir.mkdir();
            }
        	
            //reader data from the file
            File file = new File(resourcesDir, "training_data.csv");
            BufferedReader reader = new BufferedReader(new FileReader(file));
            
            //ensure file is not empty
            if (file.length() == 0) {
                System.out.println("training file empty");
                fileEmpty = true;
                return;
            }
            
            String line; //stores each line
            int lineNum = 0; //counts lines
            
            //read data line by line
            while ((line = reader.readLine()) != null) {
            	//if file empty break from look
            	if (fileEmpty) break;
            	//ignore header
            	if (line.startsWith("top1") || line.startsWith("//")) continue;
            	//split line into parts by commas
                String[] parts = line.split(",");
                
                //create a input array of features
                double[] input = new double[27];
                //loop through each feature
                for (int i = 0; i < 27; i++) {
                	//parse the string to double
                    input[i] = Double.parseDouble(parts[i]);
                }

                //make last column (movement) label and parse it
                int label = Integer.parseInt(parts[27]);
                //create array for one hot encoded output
                double[] output = new double[3];
                //set outputs based on label
                if (label == -1) output[0] = 1;
                else if (label == 0) output[1] = 1;
                else if (label == 1) output[2] = 1;
                
                //sort label to each input type
                if (label == -1) {
                    inputUp.add(input);
                    outputUp.add(output);
                    up++;
                } else if (label == 0) {
                    inputStay.add(input);
                    outputStay.add(output);
                    stay++;
                } else if (label == 1) {
                    inputDown.add(input);
                    outputDown.add(output);
                    down++;
                }

                lineNum++;
            }
            
            //output info for debugging
            System.out.println("data lines: " + lineNum);
            System.out.println("Up   (-1): " + up);
            System.out.println("Stay ( 0): " + stay);
            System.out.println("Down ( 1): " + down);
                      
            reader.close();//close file reader
            
            //find the minimum size of among input types
            int minSize = Math.min(inputUp.size(), Math.min(inputStay.size(), inputDown.size()));
            System.out.println("balancing dataset to " + minSize + " per input");

            //create new list to store balanced data
            List<double[]> balancedInputs = new ArrayList<>();
            List<double[]> balancedOutputs = new ArrayList<>();

            //combine all balanced inputs in one list per type
            balancedInputs.addAll(inputUp.subList(0, minSize));
            balancedOutputs.addAll(outputUp.subList(0, minSize));
            balancedInputs.addAll(inputStay.subList(0, minSize));
            balancedOutputs.addAll(outputStay.subList(0, minSize));
            balancedInputs.addAll(inputDown.subList(0, minSize));
            balancedOutputs.addAll(outputDown.subList(0, minSize));
            
            //create training data set
            MLDataSet trainingSet = new BasicMLDataSet();
            //loop through balanced data
            for (int i = 0; i < balancedInputs.size(); i++) {
            	//add the input and outputs to trainingset
                trainingSet.add(new BasicMLData(balancedInputs.get(i)), new BasicMLData(balancedOutputs.get(i)));
            }
            
            //train the network
            final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
            
            //store amount of epochs and lastError
            int epoch = 1;
            double lastError = Double.MAX_VALUE;
            
            System.out.println("training model");
            do {//start a training loop
            	//perform training iteration
                train.iteration();
                
                //get the error
                double error = train.getError();
                //every 10 epochs print error
                if (epoch % 10 == 0) {
                    System.out.println("Epoch #" + epoch + " Error: " + error);
                }
                
                //check for convergence stop training if so
                if (Math.abs(lastError - error) < 0.0001) {
                    break;
                }
                
                lastError = error;
                epoch++;
            } while (epoch < 1000 && train.getError() > 0.01); //train till max epoch 100 or min error of 0.01
            
            train.finishTraining();
            
            System.out.println("training complete");
            System.out.println("final error: " + train.getError());
            System.out.println("num of epochs: " + epoch);
            
        } catch (IOException e) {
            System.out.println("error loading training data: " + e.getMessage());
        }
    }
    
    //mehod to predict move for game
    public int predictMove(LinkedList<byte[]> model, int playerRow, int lastMove) {
        //Extract features using same method for collection to ensure model sees what its learned
        double[] features = DataCollection.extractFeatures(model, playerRow, lastMove);
        
        //output these features for debugging
        System.out.println("features seen by AI: " + Arrays.toString(features));
        
        //give the current game features to the model
        MLData input = new BasicMLData(features); //create input from features
        MLData output = network.compute(input); //get a output from model based on features
        
        //get the probablity for best move from model
        double[] outputArray = output.getData();
        int maxIndex = 0;
        
        //loop through the outputs 
        for (int i = 1; i < outputArray.length; i++) {
        	//if current output is higher then current its better move 
            if (outputArray[i] > outputArray[maxIndex]) {
            	//update the index
                maxIndex = i;
            }
        }
        
        //convert this index to a move value
        if (maxIndex == 0) {
            return -1; //up 
        } else if (maxIndex == 1) {
            return 0;  //stay 
        } else {
            return 1; //down
        }
    }
    
    //method to save network to file
    public void saveNetwork() {
    	try {
    		//ensure resource folder exists
    	    File resourcesDir = new File("resources");
            if (!resourcesDir.exists()) {
                resourcesDir.mkdir();
            }
            
            File networkFile = new File(resourcesDir, "trained_network.eg");
            org.encog.persist.EncogDirectoryPersistence.saveObject(
                networkFile, network);
            System.out.println("saved neural network to: " + networkFile.getPath());
            
        }
    	catch (Exception e) {
            System.out.println("error whill saving network file " + e.getMessage());
        }
    }
}