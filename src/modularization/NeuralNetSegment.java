package modularization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import ann.AccruingWeight;
import ann.ActivationFunction;
import ann.BiasNode;
import ann.Connection;
import ann.FFNeuralNetwork;
import ann.Layer;
import ann.Node;
import ann.Node.Factory;

public class NeuralNetSegment {

	public static FFNeuralNetwork createNNSegment(FFNeuralNetwork original,
		Node includedOutput, ArrayList<Node> includedInputs) {
		
		FFNeuralNetwork result = new FFNeuralNetwork();
		LinkedList<Layer<? extends Node>> resultLayers = result.getLayers();

		Map<Node, Node> seenFodeToNode = new HashMap<Node, Node>();
		for (Node fromFode : includedInputs) {
			if (fromFode == null) { // out of bounds input... put in layer but don't connect to outputs
				addNewNodeToLayer(resultLayers, 0, result.nodeFactory, ActivationFunction.LINEAR);
			} else {
				findConnToNodesLeadingTo(fromFode, includedOutput, result.nodeFactory, resultLayers, seenFodeToNode);
			}
		}
//		debug(original, result);
		return result;
	}
	
	private static void findConnToNodesLeadingTo(Node fromFode, Node toFode, Node.Factory<? extends Node> nodeFactory,
			LinkedList<Layer<? extends Node>> layers, Map<Node, Node> seenFodeToNode) {
		ArrayList<DoThingContainer> sofar = new ArrayList<DoThingContainer>();
		findConnToNodesLeadingTo(null, null, fromFode, toFode, 0, layers, nodeFactory, seenFodeToNode, sofar);
	}
	
	private static void findConnToNodesLeadingTo(Connection conn2FromFode, DoThingContainer prevDTC, Node fromFode, Node toFode,
			int depth, LinkedList<Layer<? extends Node>> layers, Node.Factory<? extends Node> nodeFactory,
			Map<Node, Node> seenFodeToNode, ArrayList<DoThingContainer> sofar) {
		ActivationFunction actFn = fromFode.getActivationFunction();
		DoThingContainer dtc = new DoThingContainer(conn2FromFode, prevDTC, fromFode, seenFodeToNode, layers, depth, nodeFactory, actFn);
		sofar.add(dtc);
		if (fromFode.getOutputConnections().isEmpty()) {
			if (fromFode == toFode) {
				for (DoThingContainer thing : sofar) thing.doit();
				sofar.clear();
			} else {
				sofar.remove(sofar.size()-1);
			}
			return;
		}
		
		for (Connection conn : fromFode.getOutputConnections()) {
			findConnToNodesLeadingTo(conn, dtc, conn.getOutputNode(), toFode, depth+1, layers, nodeFactory,
					seenFodeToNode, sofar);
		}
	}
	
	private static Layer<? extends Node> getOrCreateLayer(LinkedList<Layer<? extends Node>> layers, int depth,
			Node.Factory<? extends Node> nodeFactory, ActivationFunction activationFunction) {
		if (layers.size() <= depth) {
			layers.add(Layer.create(0, activationFunction, nodeFactory));
		}
		return layers.get(depth);
	}
	
	public static void main(String[] args) {
		test();
	}

	private static void test() {
		FFNeuralNetwork fullANN = new FFNeuralNetwork(ActivationFunction.SIGMOID0p5, 3, 3, new int[] {5});
		ArrayList<? extends Node> inputNodes = fullANN.getInputNodes();
		ArrayList<? extends Node> outputNodes = fullANN.getOutputNodes();
		ArrayList<Node> usedInputs = new ArrayList<Node>();
		usedInputs.add(inputNodes.get(1));
		usedInputs.add(inputNodes.get(2));
		Node usedOutput = outputNodes.get(2);
		
		FFNeuralNetwork segmentANN = createNNSegment(fullANN, usedOutput, usedInputs);
		debug(fullANN, segmentANN);
	}
	
	private static void debug(FFNeuralNetwork fullANN, FFNeuralNetwork segmentANN) {
		System.out.println("FULL\n");
		fullANN.printWeights();
		System.out.println("SEGMENT\n");
		segmentANN.printWeights();
	}
	
	private static Node addNewNodeToLayer(LinkedList<Layer<? extends Node>> layers, int depth,
			Factory<? extends Node> nodeFactory, ActivationFunction activationFunction) {
		Layer<? extends Node> layer = getOrCreateLayer(layers, depth, nodeFactory, activationFunction);
		Node newNode = nodeFactory.create(activationFunction, layer, String.valueOf(layer.getNodes().size()));
		layer.addNode(newNode);
		return newNode;
	}
	
	private static class DoThingContainer {
		private final Connection conn2FromFode;
		private final DoThingContainer prevDTC;
		private final Node fromFode;
		private final Map<Node, Node> seenFodeToNode;
		private final LinkedList<Layer<? extends Node>> layers;
		private final int depth;
		private final Node.Factory<? extends Node> nodeFactory;
		private final ActivationFunction activationFunction;
		private Node newNode;

		public DoThingContainer(Connection conn2FromFode, DoThingContainer prevDTC, Node fromFode,
				Map<Node, Node> seenFodeToNode, LinkedList<Layer<? extends Node>> layers,
				int depth, Factory<? extends Node> nodeFactory, ActivationFunction activationFunction) {
			this.conn2FromFode = conn2FromFode;
			this.prevDTC = prevDTC;
			this.fromFode = fromFode;
			this.seenFodeToNode = seenFodeToNode;
			this.layers = layers;
			this.depth = depth;
			this.nodeFactory = nodeFactory;
			this.activationFunction = activationFunction;
		}
		
		void doit() {
			newNode = seenFodeToNode.get(fromFode);
			if (newNode == null) {
				newNode = addNewNodeToLayer(layers, depth, nodeFactory, activationFunction);
				if (depth > 0) {
					final double biasW = BiasNode.getBiasConnection(fromFode).getWeight().getWeight();
					BiasNode.connectToNode(newNode, new AccruingWeight(biasW));
				}
				seenFodeToNode.put(fromFode, newNode);
			}
			if (prevDTC != null) {
				Connection conn = Connection.getOrCreate(prevDTC.newNode, newNode);
				conn.getWeight().setWeight(conn2FromFode.getWeight().getWeight());
				conn.getWeight().enactWeightChange(0);
			}
		}
	}
}
