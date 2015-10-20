package modularization;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import ann.Connection;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;
import ann.indirectencodings.IndirectInput;
import ann.indirectencodings.RelationManager;
import ann.testing.GridGame;
import modeler.ModelLearner;

/**
 * Given the key of some output variable, it looks up the keys to related input variables.
 * Then, given the values of those inputs for some sample, it generates the value for the output.
 *
 */
public class ReusableModule {
	
	private final Collection<IndirectInput> relations = new ArrayList<IndirectInput>();
	private final Map<IndirectInput, Node> rel2ModuleNode = new HashMap<IndirectInput, Node>();

	private FFNeuralNetwork neuralNet;
	
	/**
	 * creates a reusable module
	 * @param relMngr 
	 */
	public static ReusableModule createNeighborHoodModule(GridGame game, ModelLearner modeler,
			RelationManager relMngr, FFNeuralNetwork fullModelNN, Node outputOfInterest) {
		ReusableModule result = new ReusableModule();
		result.relations.addAll(RelationManager.USED_RELS);
		result.setModuleNN(fullModelNN, outputOfInterest, relMngr);
		return result;
	}

	/**
	 * crops out section of full model NN that corresponds to the output of interest and its relations
	 */
	private void setModuleNN(FFNeuralNetwork fullModelNN, Node outputOfInterest, RelationManager relMngr) {
		ArrayList<Node> inputsOfInterest = getRelNodes(outputOfInterest, relMngr);
		neuralNet = ModularizationUtils.createNNSegment(fullModelNN, outputOfInterest, inputsOfInterest);
		if (rel2ModuleNode.isEmpty()) {
			ArrayList<? extends Node> inputNodes = neuralNet.getInputNodes();
			int i = 0;
			for (IndirectInput rel : relations) rel2ModuleNode.put(rel, inputNodes.get(i++));
		}
	}

	public double evaluateOutput(Node output, RelationManager relMngr) {
		Collection<Node> relNodes = getRelNodes(output, relMngr);
		final double[] activations = new double[relNodes.size()];
		int i = 0;
		for (Node input : relNodes) {
			activations[i++] = input != null ? input.getActivation() : 0;
		}
		FFNeuralNetwork.feedForward(neuralNet.getInputNodes(), activations);
		final double result = neuralNet.getOutputNodes().get(0).getActivation();
		return result;
	}
	
	private ArrayList<Node> getRelNodes(Node output, RelationManager relMngr) {
		ArrayList<Node> relNodes = new ArrayList<Node>();
		for (IndirectInput rel : relations) {
			Node relNode = relMngr.getRelNode(output, rel);
			relNodes.add(relNode);
		}
		return relNodes;
	}
	
	@Override
	public String toString() {
		if (rel2ModuleNode.isEmpty()) return "uninitialized module";
		StringBuilder sb = new StringBuilder();
		for (IndirectInput ii : relations) {
			Node node = rel2ModuleNode.get(ii);
			if (node == null) continue;
			Collection<Connection> outputConnections = node.getOutputConnections();
			if (!outputConnections.isEmpty()) {
				sb.append(ii.toString().substring(0, 1));
				sb.append(Utils.round(outputConnections.iterator().next().getWeight().getWeight(), 2));
				sb.append(" ");
			}
		}
		return sb.toString();
	}

	public FFNeuralNetwork getNeuralNet() {
		return neuralNet;
	}
}
