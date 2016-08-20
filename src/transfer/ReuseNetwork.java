package transfer;

import java.util.*;

import ann.*;

@SuppressWarnings("serial")
public class ReuseNetwork extends FFNeuralNetwork {
	
	private final Collection<FFNeuralNetwork> sources = new ArrayList<FFNeuralNetwork>();

	private ReuseNetwork(FFNeuralNetwork... sources) {
		for (FFNeuralNetwork source : sources) this.sources.add(source);
	}

	/** fully connected with source as hidden layer 
	 * only works with input and output layers equal size between target and source */
	public static ReuseNetwork createSandwichedNetwork(FFNeuralNetwork source, boolean frieze) {
		ArrayList<? extends Node> sourceInputs = source.getInputNodes();
		ArrayList<? extends Node> sourceOutputs = source.getOutputNodes();
		ActivationFunction actFn = sourceInputs.get(0).getActivationFunction();
		if (frieze) {
			Collection<Connection> conns = Connection.getAllConnections(source);
			for (Connection conn : conns) conn.getWeight().frieze();
		}
		ReuseNetwork result = new ReuseNetwork(source);
		Layer<? extends Node> inputLayer = Layer.createInputLayer(sourceInputs.size(), result.nodeFactory);
		result.getLayers().add(inputLayer);
		Layer.fullyConnect(inputLayer.getNodes(), sourceInputs);
//		for (Node n : sourceInputs) n.setActivationFunction(actFn);
		Layer<? extends Node> outputLayer = Layer.createHiddenFromInputLayer(sourceOutputs,
				sourceOutputs.size(), actFn, result.nodeFactory);
		result.getLayers().add(outputLayer);
		result.getBiasNode().connectToLayer(outputLayer);
		return result;
	}
	
	// TODO
	// source network may be different size from target
}
