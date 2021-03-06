package ann;

import java.io.Serializable;
import java.util.*;

@SuppressWarnings("serial")
public class Node implements Serializable {
	
	protected final Layer<? extends Node> parentLayer;
	protected ActivationFunction activationFunction;
	protected String nodeInLayer;
	protected Double preactivation;
	protected double activation;
	protected ArrayList<Connection> inputConnections;
	protected ArrayList<Connection> outputConnections;
	
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

	public ArrayList<Connection> getInputConnections() {
		return inputConnections;
	}

	public ArrayList<Connection> getOutputConnections() {
		return outputConnections;
	}
	
	@Override
	public String toString() {
		return shortId();
//		return parentLayer.toString() + nodeInLayer + " A:" + activation;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
	
	public void setName(String name) {
		this.nodeInLayer = name;
	}

	public interface Factory<N extends Node> extends Serializable {
		public N create(ActivationFunction activationFunction, Layer<? extends Node> parentLayer, String nodeInLayer);
	}
	
	public static Node.Factory<? extends Node> BASIC_NODE_FACTORY = new Node.Factory<Node>() {
		@Override
		public Node create(ActivationFunction activationFunction,
				Layer<? extends Node> parentLayer, String nodeInLayer) {
			return new Node(activationFunction, parentLayer, nodeInLayer);
		}
	};

	public void setActivationFunction(ActivationFunction actFn) {
		this.activationFunction = actFn;
	}

	public String shortId() {
		return parentLayer.toString() + nodeInLayer;
	}
}
