package transfer;

import java.util.ArrayList;

import ann.ActivationFunction;
import ann.Connection;
import ann.FFNeuralNetwork;
import ann.Node;
import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;

/**
 * A set of connections from a target network's input nodes TI
 * to a source network's input nodes SI and another set of
 * connections from the source network's output nodes SO to
 * the target network's output nodes TO.
 * Each connection between TI_j and SI_k shares a weight with
 * the connection between SO_k and TO_j.
 */
public class IOVarMapper {

	private FFNeuralNetwork sourceNet;
	private FFNeuralNetwork targetNet;
	private ArrayList<Connection> inConns = new ArrayList<Connection>();
	private ArrayList<Connection> outConns = new ArrayList<Connection>();

	public IOVarMapper(FFNeuralNetwork sourceNet, FFNeuralNetwork targetNet) {
		this.sourceNet = sourceNet;
		this.targetNet = targetNet;
		createConnections();
	}
	
	private void createConnections() {
		ArrayList<? extends Node> trgIn = targetNet.getInputNodes();
		ArrayList<? extends Node> srcIn = sourceNet.getInputNodes();
		ArrayList<? extends Node> trgOut = targetNet.getOutputNodes();
		ArrayList<? extends Node> srcOut = sourceNet.getOutputNodes();
		int inN = trgIn.size();
		int outN = trgOut.size();
		if (inN != trgIn.size() || outN != srcOut.size()) // this req can be removed later
			throw new IllegalStateException("SRC & TRG must have same output size");
		if (inN < outN) throw new IllegalStateException("input cannot be smaller than output");
		// states
		for (int i = 0; i < outN; i++) {
			for (int j = 0; j < outN; j++) {
				Connection inC = Connection.getOrCreate(trgIn.get(i), srcIn.get(j));
				Connection outC = Connection.getOrCreate(srcOut.get(j), trgOut.get(i), inC.getWeight());
				inConns.add(inC);
				outConns.add(outC);
			}
		}
		// actions
		for (int i = outN; i < inN; i++) {
			for (int j = outN; j < inN; j++) {
				Connection inC = Connection.getOrCreate(trgIn.get(i), srcIn.get(j));
				inConns.add(inC);
			}
		}
	}
	
	public static ModelLearner createReuseNetModel(ModelLearner src, int turns) {
		ModelLearnerHeavy trg = new ModelLearnerHeavy(500, new int[] {}, null, new int[] {},
				ActivationFunction.SIGMOID0p5, turns);
		return null; // TODO
	}
}
