package deepnets;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.*;

import deepnets.convolution.*;

@SuppressWarnings("serial")
public class FFNeuralNetwork implements NeuralNetwork, Serializable {
	
	private final LinkedList<Layer<? extends Node>> layers = new LinkedList<Layer<? extends Node>>();
	public final Node.Factory<? extends Node> nodeFactory;
	
	public FFNeuralNetwork() {
		this.nodeFactory = Node.BASIC_NODE_FACTORY;
	}
	
	public FFNeuralNetwork(ActivationFunction actFn, int numInputs, int numOutputs,
			Node.Factory<? extends Node> nodeFactory, int... numHidden) {
		this.nodeFactory = nodeFactory;
		Layer<? extends Node> inputLayer = Layer.createInputLayer(numInputs, nodeFactory);
		layers.add(inputLayer);
		Layer<? extends Node> lastLayer = inputLayer;
		for (int h : numHidden) {
			Layer<? extends Node> hiddenLayer = Layer.createHiddenFromInputLayer(lastLayer, h, actFn, nodeFactory);
			BiasNode.connectToLayer(hiddenLayer);
			lastLayer = hiddenLayer;
			layers.add(lastLayer);
		}
		Layer<? extends Node> outputLayer = Layer.createHiddenFromInputLayer(lastLayer, numOutputs, actFn, nodeFactory);
		layers.add(outputLayer);
		BiasNode.connectToLayer(outputLayer);
	}
	public FFNeuralNetwork(ActivationFunction actFn, int numInputs, int numOutputs, int... numHidden) {
		this(actFn, numInputs, numOutputs, Node.BASIC_NODE_FACTORY, numHidden);
	}
	protected Node.Factory<? extends Node> getNodeFactory() {
		return nodeFactory;
	}
	
	@Override
	public ArrayList<? extends Node> getInputNodes() {
		return layers.get(0).getNodes();
	}
	@Override
	public ArrayList<? extends Node> getOutputNodes() {
		return layers.get(layers.size()-1).getNodes();
	}
	
	public Collection<Layer<? extends Node>> getLayers() {
		return layers;
	}
	public int getNumLayers() {
		return layers.size();
	}
	public int getLayerSize(int layer) {
		return layers.get(layer).getNodes().size();
	}

	public void addNode(int l, ActivationFunction actFn, AccruingWeight biasWeight) {
		Layer<? extends Node> layer = layers.get(l);
		Node node = getNodeFactory().create(actFn, layer, null);
		layer.addNode(node);
		int numLayers = layers.size();
		if (l < numLayers - 1) {
			Layer<? extends Node> nextLayer = layers.get(l + 1);
			for (Node next : nextLayer.getNodes()) Connection.getOrCreate(node, next);
		}
		if (l > 0) {
			Layer<? extends Node> prevLayer = layers.get(l - 1);
			for (Node prev : prevLayer.getNodes()) Connection.getOrCreate(prev, node);
		}
		// bias null check cuz dont wanna connect input to bias
		if (biasWeight != null) BiasNode.connectToNode(node, biasWeight);
	}

	public static double stdError(Collection<? extends Node> inputNodes,
			Collection<? extends Node> outputNodes, Collection<DataPoint> data) {
		double sumErr = 0;
		for (DataPoint dp : data) {
			feedForward(inputNodes, dp.getInput());
			double error = getError(dp.getOutput(), outputNodes);
			sumErr += error;
		}
		return Math.sqrt(sumErr / data.size());
	}
	public static double getError(double[] target, Collection<? extends Node> outputNodes) {
		double error = 0;
		for (int i = 0; i < target.length; i++) {
			double output = i < outputNodes.size()
					? ((ArrayList<? extends Node>)outputNodes).get(i).getActivation() : 0;
			error += Math.pow(target[i] - output, 2);
//			if (i == 1) System.out.println(target[i] +"	-	"+ output);
		}
		return error / target.length;
	}
	public static void feedForward(Collection<? extends Node> nodes, double... clampInputs) {
		feedForward(nodes, null, clampInputs);
	}
	public static void feedForward(Collection<? extends Node> nodes, 
			Collection<? extends Node> lastNodes, double... clampInputs) {
		Collection<Node> nextNodes = new HashSet<Node>();
		int i = 0;
		for (Node n : nodes) {
			if (i >= nodes.size()) break;
			if (clampInputs != null) n.clamp(i < clampInputs.length ? clampInputs[i++] : 0);
			else n.preactivate(); // parallely
			Collection<Connection> ocs = n.getOutputConnections();
			for (Connection conn : ocs) nextNodes.add(conn.getOutputNode());
		}
		if (clampInputs == null) for (Node n : nodes) n.activate(); // parallely
		if (lastNodes != null) nextNodes.removeAll(lastNodes); // avoid infinite loop
		if (!nextNodes.isEmpty()) feedForward(nextNodes, nodes, null);
	}
	
	public static void feedForward(ConvolutionNetwork cnn, BufferedImage img) {
		cnn.getInputLayer().clamp(img);
		for (ConvolutionLayer cl : cnn.getHiddenLayers(0)) feedForward(cl.getNodes(), null);
	}
	
	public static void backPropagate(Collection<? extends Node> nodes,
			double learningRate, double momentum, double wgtDecay, double... targets) {
		// TODO stochasticity 
		Collection<Node> nextNodes = new HashSet<Node>();
		Collection<AccruingWeight> weights = new ArrayList<AccruingWeight>();
		int i = 0;
		for (Node n : nodes) {
			if (targets != null && i >= targets.length) break;
			final double activation = n.getActivation();
			final double derivative = n.getActivationFunction().derivative(activation);
			double delta = derivative;
			if (targets != null) delta *= (targets[i++] - activation);
			else{
				double sumBlame = 0;
				for (Connection outConn : n.getOutputConnections()) sumBlame += outConn.getWeight().getBlameFromOutput();
				delta *= sumBlame;
			}
			Collection<Connection> ics = n.getInputConnections();
			for (Connection conn : ics) {
				final Node inNode = conn.getInputNode();
				final AccruingWeight w = conn.getWeight();
				w.propagateError(delta, inNode.getActivation(), learningRate, momentum, false);
				weights.add(w);
				nextNodes.add(inNode);
			}
		}
		for (AccruingWeight w : weights) w.enactWeightChange(wgtDecay);
		if (!nextNodes.isEmpty()) backPropagate(nextNodes, learningRate, momentum, wgtDecay, null);
	}
	
	public void report(Collection<DataPoint> data) {
		for (DataPoint dp : data) {
			FFNeuralNetwork.feedForward(getInputNodes(), dp.getInput());
			report();
		}
		System.out.println("E: " + FFNeuralNetwork.stdError(getInputNodes(), getOutputNodes(), data));
	}
	public void report() {
		String printy = "inputs:	";
		for (Node n : getInputNodes()) printy += n.getActivation() + " ";
		printy += "	outputs:	";
		for (Node n : getOutputNodes()) printy += n.getActivation() + " ";
		System.out.println(printy);
	}
	
}
