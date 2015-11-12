package modulemanagement;

import java.util.ArrayList;
import java.util.Collection;

import ann.FFNeuralNetwork;
import ann.Node;
import ann.indirectencodings.IndirectInput;
import ann.indirectencodings.RelationManager;
import modeler.ModelLearner;
import modularization.ModularizationUtils;

/**
 * Given the key of some output variable, it looks up the keys to related input variables.
 * Then, given the values of those inputs for some sample, it generates the value for the output.
 *
 */
@SuppressWarnings("serial")
public class ReusableNNModule extends ReusableModule<Node> {
	
	/**
	 * creates a reusable module
	 * @param relMngr 
	 */
	public static ReusableNNModule createNeighborHoodModule(ModelLearner modeler,
			RelationManager<Node> relMngr, FFNeuralNetwork fullModelNN, Node outputOfInterest) {
		ReusableNNModule result = new ReusableNNModule();
		result.relations.addAll(relMngr.getUsedRels());
		result.setModuleNN(fullModelNN, outputOfInterest, relMngr);
		return result;
	}

	/**
	 * crops out section of full model NN that corresponds to the output of interest and its relations
	 */
	private void setModuleNN(FFNeuralNetwork fullModelNN, Node outputOfInterest, RelationManager<Node> relMngr) {
		ArrayList<Node> inputsOfInterest = getRelInputs(outputOfInterest, relMngr);
		neuralNet = ModularizationUtils.createNNSegment(fullModelNN, outputOfInterest, inputsOfInterest);
		if (rel2ModuleInput.isEmpty()) {
			ArrayList<? extends Node> inputNodes = neuralNet.getInputNodes();
			int i = 0;
			for (IndirectInput rel : relations) rel2ModuleInput.put(rel, inputNodes.get(i++));
		}
	}

	@Override
	public double evaluateOutput(Node output, RelationManager<Node> relMngr, double[] fullInput) {
		return evaluateOutput(output, relMngr);
	}
	public double evaluateOutput(Node output, RelationManager<Node> relMngr) {
		Collection<Node> relNodes = getRelInputs(output, relMngr);
		final double[] activations = new double[relNodes.size()];
		int i = 0;
		for (Node input : relNodes) {
			activations[i++] = input != null ? input.getActivation() : 0;
		}
		FFNeuralNetwork.feedForward(neuralNet.getInputNodes(), activations);
		final double result = neuralNet.getOutputNodes().get(0).getActivation();
		return result;
	}

	@Override
	protected int getVectorKey(Node key) {
		throw new IllegalStateException("unused");
	}

}
