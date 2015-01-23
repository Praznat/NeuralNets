package deepnets;

import java.util.*;

public class Node {
	
	private final ActivationFunction activationFunction;
	private final Layer<? extends Node> parentLayer;
	private final String nodeInLayer;
	private double activation;
	private final Collection<Connection> inputConnections = new ArrayList<Connection>();
	private final Collection<Connection> outputConnections = new ArrayList<Connection>();
	
	public Node(ActivationFunction activationFunction, Layer<? extends Node> parentLayer, String nodeInLayer) {
		this.activationFunction = activationFunction;
		this.parentLayer = parentLayer;
		this.nodeInLayer = nodeInLayer;
	}

	public double getActivation() {
		return this.activation;
	}
	public void activate() {
		double sumIn = 0;
		for (Connection ic : inputConnections) sumIn += ic.getWeight().getWeight() * ic.getInputNode().getActivation();
		this.activation = activationFunction.feed(sumIn);
	}
	
	/** forces an activation of given value */
	public void clamp(double value) {
		this.activation = value;
	}

	public Collection<Connection> getInputConnections() {
		return inputConnections;
	}

	public Collection<Connection> getOutputConnections() {
		return outputConnections;
	}
	
	@Override
	public String toString() {
		return parentLayer.toString() + nodeInLayer + " A:" + activation;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

}
