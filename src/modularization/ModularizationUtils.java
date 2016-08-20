package modularization;

import java.util.ArrayList;
import java.util.Collection;
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
import ann.indirectencodings.IndirectInput;
import ann.indirectencodings.RelationManager;

public class ModularizationUtils {
	
	/**
	 * reinitialize internal structure of a given ann, only creating connections
	 * between inputs and outputs with relationships specified by relMngr,
	 * with a single layer of hidden nodes of size numHidden between each output
	 * and its related set of inputs
	 * @param ann
	 * @param relMngr
	 * @param numHidden
	 */
	public static void initializeANNOnlyConnectingRelatedVars(FFNeuralNetwork ann,
			RelationManager relMngr, int[] hiddenPerOutput) {
		ArrayList<? extends Node> inputs = ann.getInputNodes();
		ArrayList<? extends Node> outputs = ann.getOutputNodes();
		for (Node n : inputs) n.getOutputConnections().clear();
		for (Node n : outputs) {
			n.getInputConnections().clear();
			ann.getBiasNode().disconnectFrom(n);
		}
		LinkedList<Layer<? extends Node>> layers = ann.getLayers();
		int numLayers = ann.getNumLayers();
		for (int i = numLayers - 2; i >= 1; i--) {
			Layer<? extends Node> removed = layers.remove(i);
			for (Node n : removed.getNodes()) ann.getBiasNode().disconnectFrom(n);
		}
		ActivationFunction actFn = inputs.get(0).getActivationFunction();
		for (int i = 0; i < hiddenPerOutput.length; i++) {
			layers.add(i+1, Layer.create(0, actFn, ann.nodeFactory));
		}
		for (Node output : outputs) {
			Collection<Node> outputColl = new ArrayList<Node>();
			outputColl.add(output);
			ann.getBiasNode().connectToNode(output);
			Collection<Node> lastNodes = new ArrayList<Node>();
			for (Node input : inputs) {
				IndirectInput rel = relMngr.getRel(input, output);
				if (relMngr.getUsedRels().contains(rel)) lastNodes.add(input);
			}
			for (int h = 0; h < hiddenPerOutput.length; h++) {
				final int hidden = hiddenPerOutput[h];
				Layer<? extends Node> layer = layers.get(h+1);
				Collection<Node> newNodes = new ArrayList<Node>();
				for (int i = 0; i < hidden; i++) {
					Node n = ann.nodeFactory.create(actFn, layer, String.valueOf(i));
					layer.addNode(n);
					newNodes.add(n);
					ann.getBiasNode().connectToNode(n);
				}
				Layer.fullyConnect(lastNodes, newNodes);
				lastNodes.clear();
				lastNodes.addAll(newNodes);
			}
			Layer.fullyConnect(lastNodes, outputColl);
		}
	}

	/**
	 * returns new neural network with new nodes and connections having same structure
	 * and weights as original network but only between given includedInputs and includedOutput
	 * @param original
	 * @param includedOutput
	 * @param includedInputs
	 * @return
	 */
	public static FFNeuralNetwork createNNSegment(FFNeuralNetwork original,
		Node includedOutput, ArrayList<Node> includedInputs) {
		
		FFNeuralNetwork result = new FFNeuralNetwork();
		LinkedList<Layer<? extends Node>> resultLayers = result.getLayers();

		Map<Node, Node> seenFodeToNode = new HashMap<Node, Node>();
		for (Node fromFode : includedInputs) {
			if (fromFode == null) { // out of bounds input... put in layer but don't connect to outputs
				addNewNodeToLayer(resultLayers, 0, result.nodeFactory, ActivationFunction.LINEAR);
			} else {
				findConnToNodesLeadingTo(fromFode, includedOutput, result.nodeFactory, resultLayers, seenFodeToNode,
						original.getBiasNode(), result.getBiasNode());
			}
		}
//		debug(original, result);
		return result;
	}
	
	private static void findConnToNodesLeadingTo(Node fromFode, Node toFode, Node.Factory<? extends Node> nodeFactory,
			LinkedList<Layer<? extends Node>> layers, Map<Node, Node> seenFodeToNode, BiasNode oldBias, BiasNode newBias) {
		ArrayList<DoThingContainer> sofar = new ArrayList<DoThingContainer>();
		findConnToNodesLeadingTo(null, null, fromFode, toFode, 0, layers, nodeFactory, seenFodeToNode, sofar, oldBias, newBias);
	}
	
	private static void findConnToNodesLeadingTo(Connection conn2FromFode, DoThingContainer prevDTC, Node fromFode, Node toFode,
			int depth, LinkedList<Layer<? extends Node>> layers, Node.Factory<? extends Node> nodeFactory,
			Map<Node, Node> seenFodeToNode, ArrayList<DoThingContainer> sofar, BiasNode oldBias, BiasNode newBias) {
		ActivationFunction actFn = fromFode.getActivationFunction();
		DoThingContainer dtc = new DoThingContainer(conn2FromFode, prevDTC, fromFode, seenFodeToNode, layers, depth, nodeFactory,
				actFn, oldBias, newBias);
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
					seenFodeToNode, sofar, oldBias, newBias);
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
		private final BiasNode oldBiasNode;
		private final BiasNode newBiasNode;
		private Node newNode;

		public DoThingContainer(Connection conn2FromFode, DoThingContainer prevDTC, Node fromFode,
				Map<Node, Node> seenFodeToNode, LinkedList<Layer<? extends Node>> layers,
				int depth, Factory<? extends Node> nodeFactory, ActivationFunction activationFunction,
				BiasNode oldBiasNode, BiasNode newBiasNode) {
			this.conn2FromFode = conn2FromFode;
			this.prevDTC = prevDTC;
			this.fromFode = fromFode;
			this.seenFodeToNode = seenFodeToNode;
			this.layers = layers;
			this.depth = depth;
			this.nodeFactory = nodeFactory;
			this.activationFunction = activationFunction;
			this.oldBiasNode = oldBiasNode;
			this.newBiasNode = newBiasNode;
		}
		
		void doit() {
			newNode = seenFodeToNode.get(fromFode);
			if (newNode == null) {
				newNode = addNewNodeToLayer(layers, depth, nodeFactory, activationFunction);
				if (depth > 0) {
					final double biasW = oldBiasNode.getBiasConnection(fromFode).getWeight().getWeight();
					newBiasNode.connectToNode(newNode, new AccruingWeight(biasW));
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
