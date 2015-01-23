package deepnets;

import java.util.Collection;


public interface NeuralNetwork {
	public Collection<? extends Node> getInputNodes();

	public Collection<? extends Node> getOutputNodes();
}
