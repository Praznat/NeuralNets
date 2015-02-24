package deepnets;

import java.util.*;

public class Node {
	
	private final ActivationFunction activationFunction;
	private final Layer<? extends Node> parentLayer;
	private String nodeInLayer;
	private Double preactivation;
	private double activation;
	protected Collection<Connection> inputConnections;
	protected Collection<Connection> outputConnections;
	
	public Node(ActivationFunction activationFunction, Layer<? extends Node> parentLayer, String nodeInLayer) {
		this.activationFunction = activationFunction;
		this.parentLayer = parentLayer;
		this.nodeInLayer = nodeInLayer != null ? nodeInLayer : String.valueOf(parentLayer.getNodes().size());
		inputConnections = new ArrayList<Connection>();
		outputConnections = new ArrayList<Connection>();
	}

	public double getActivation() {
		return this.activation;
	}
	public double calculateActivation() {
		double sumIn = 0;
		for (Connection ic : inputConnections) sumIn += ic.getWeight().getWeight() * ic.getInputNode().getActivation();
		return activationFunction.feed(sumIn);
	}
	public void activate() {
		if (preactivation != null) { // if youve preactivated use that
			this.activation = this.preactivation;
			this.preactivation = null;
		} else {
			this.activation = calculateActivation(); // else just activate normally
		}
	}
	public void preactivate() {
		this.preactivation = calculateActivation();
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
	
	public void setName(String name) {
		this.nodeInLayer = name;
	}

	public interface Factory<N extends Node> {
		public N create(ActivationFunction activationFunction, Layer<? extends Node> parentLayer, String nodeInLayer);
	}
	
	public static Node.Factory<? extends Node> BASIC_NODE_FACTORY = new Node.Factory<Node>() {
		@Override
		public Node create(ActivationFunction activationFunction,
				Layer<? extends Node> parentLayer, String nodeInLayer) {
			return new Node(activationFunction, parentLayer, nodeInLayer);
		}
	};
}
