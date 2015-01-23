package deepnets;

import java.util.ArrayList;

public class BiasNode {

	public static final Node INSTANCE = ((ArrayList<Node>)Layer.createInputLayer(1).getNodes()).get(0);
	
	public static void connectToLayer(Layer<? extends Node> outputLayer) {
		for (Node n : outputLayer.getNodes()) connectToNode(n);
	}

	public static void connectToNode(Node node) {
		Connection.getOrCreate(INSTANCE, node);
	}
	public static void connectToNode(Node node, AccruingWeight weight) {
		Connection.getOrCreate(INSTANCE, node, weight);
	}
	
	public static void clearConnections() {
		INSTANCE.getOutputConnections().clear();
	}
	
}
