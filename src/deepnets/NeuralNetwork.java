package deepnets;

import java.util.ArrayList;


public interface NeuralNetwork {
	public ArrayList<? extends Node> getInputNodes();

	public ArrayList<? extends Node> getOutputNodes();
}
