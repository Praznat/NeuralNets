package ann.indirectencodings;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import ann.Connection;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;
import ann.testing.IGridGame;
import modeler.ModelLearner;

public class RelationManager<T> {
	
	private List<double[]> actionChoices;
	
	public RelationManager(List<double[]> actionChoices) {
		this.actionChoices = actionChoices;
	}
	
	private final Map<T, IndirectInput> nodeMap = new HashMap<T, IndirectInput>();
	private final Map<NodePair<T>, IndirectInput> relationMap = new HashMap<NodePair<T>, IndirectInput>();
	private final Map<T, Map<IndirectInput, T>> out2InNodeMap = new HashMap<T, Map<IndirectInput, T>>();
	private int actionClass = -1;
	private IndirectInput nothing;
	private IndirectInput right;
	private IndirectInput left;
	private IndirectInput above;
	private IndirectInput below;
	private IndirectInput same;
	private IndirectInput[] collisions;
	private IndirectInput[] actions;
	private Collection<IndirectInput> usedRels;
	
	/** 
	 * produces an input vector to the CPPN based on the node and connection characteristics 
	 */
	public double[] encodeConnection(Connection conn) {
		Node inNode = conn.getInputNode();
		Node outNode = conn.getOutputNode();
		IndirectInput inV = nodeMap.get(inNode);
		IndirectInput outV = nodeMap.get(outNode);
		IndirectInput relV = relationMap.get(new NodePair<Node>(inNode, outNode));
		if (inV == null || outV == null || relV == null) return new double[] {};
		return Utils.concat(inV.getVector(), outV.getVector(), relV.getVector());
	}
	
	public int getInputSize() {
		return Utils.concat(nodeMap.values().iterator().next().getVector(),
				relationMap.values().iterator().next().getVector()).length;
	}

	public static RelationManager<Integer> createFromGridGamePredictor(IGridGame game, int inN, int outN) {
		return createFromGridGamePredictor(game.getRows(), game.getCols(), game.getActionChoices(), inN, outN);
	}
	public static RelationManager<Integer> createFromGridGamePredictor(int nrows, int ncols, List<double[]> actionChoices,
			int inN, int outN) {
		ArrayList<Integer> inputNames = new ArrayList<Integer>();
		ArrayList<Integer> outputNames = new ArrayList<Integer>();
		for (int i = 0; i < inN; i++) inputNames.add(i);
		for (int i = 0; i < outN; i++) outputNames.add(i);
		return createFromGridGamePredictor(nrows, ncols, actionChoices, inputNames, outputNames);
	}
	public static RelationManager<Node> createFromGridGamePredictor(IGridGame game, ModelLearner modeler) {
		return createFromGridGamePredictor(game.getRows(), game.getCols(), game.getActionChoices(), modeler);
	}
	public static RelationManager<Node> createFromGridGamePredictor(int nrows, int ncols, List<double[]> actionChoices,
			ModelLearner modeler) {
		FFNeuralNetwork predictor = modeler.getTransitionsModule().getNeuralNetwork();
		ArrayList<? extends Node> inputNodes = predictor.getInputNodes();
		ArrayList<? extends Node> outputNodes = predictor.getOutputNodes();
		return createFromGridGamePredictor(nrows, ncols, actionChoices, inputNodes, outputNodes);
	}
	public static <T> RelationManager<T> createFromGridGamePredictor(int nrows, int ncols, List<double[]> actionChoices,
			ArrayList<? extends T> inputs, ArrayList<? extends T> outputs) {
		RelationManager<T> result = new RelationManager<T>(actionChoices);
		Map<T, Integer> cols = new HashMap<T, Integer>();
		Map<T, Integer> rows = new HashMap<T, Integer>();
		Map<T, Integer> objClass = new HashMap<T, Integer>();


		mapNodesToGeoSpace(cols, rows, objClass, inputs, ncols, nrows);
		mapNodesToGeoSpace(cols, rows, objClass, outputs, ncols, nrows);
		int numObjClasses = (new HashSet<Integer>(objClass.values())).size();
		result.determineRelVectors(numObjClasses);
		
		int numO = (int) Math.ceil(((double)inputs.size()) / (ncols * nrows));
		
		for (T out : outputs) {
			int outC = cols.get(out);
			int outR = rows.get(out);
			int outO = objClass.get(out);
			result.nodeMap.put(out, new IndirectInput(outC+","+outR+","+outR,
					geoToVector(outC, ncols, outR, ncols, outO, numO)));
			Map<IndirectInput, T> iiToNode = new HashMap<IndirectInput, T>();
			result.out2InNodeMap.put(out, iiToNode);
			for (T in : inputs) {
				int inC = cols.get(in);
				int inR = rows.get(in);
				int inO = objClass.get(in);
				NodePair<T> pair = new NodePair<T>(in, out);
				IndirectInput rel = result.nothing; // rel means out is REL of in
				result.nodeMap.put(in, new IndirectInput(inC+","+inR+":"+inO,
						geoToVector(inC, ncols, inR, ncols, inO, numO)));
				if (inO == result.actionClass ) {
					rel = result.actions[inC];
				} else if (inC == outC && inR == outR) {
					if (inO == outO) rel = result.same;
					else rel = result.collisions[inO];
				} else if (inO == outO) {
					if (inC == outC) {
						if (outR == inR + 1) rel = result.below;
						else if (outR == inR - 1) rel = result.above;
					} else if (inR == outR) {
						if (outC == inC + 1) rel = result.right;
						else if (outC == inC - 1) rel = result.left;
					}
				}
				if (rel != result.nothing) {
					result.relationMap.put(pair, rel);
					iiToNode.put(rel, in);
				}
			}
		}
		if (result.nodeMap.isEmpty()) throw new IllegalStateException("Empty Relation Manager Created");
		return result;
	}
	
	private void determineRelVectors(int numObjClasses) {
		actionClass = numObjClasses - 1;
		int vectLen = 2 + 5 + numObjClasses + 1 + this.actionChoices.size(); // coords + directions + objects + same + actions
		int i = 2;
		nothing = new IndirectInput("NOTHING", i++, vectLen);
		right = new IndirectInput("RIGHT", i++, vectLen);
		left = new IndirectInput("LEFT", i++, vectLen);
		above = new IndirectInput("ABOVE", i++, vectLen);
		below = new IndirectInput("BELOW", i++, vectLen);
		
		same = new IndirectInput("SAME", i++, vectLen);
		collisions = new IndirectInput[actionClass];
		for (int c = 0; c < collisions.length; c++) collisions[c] = new IndirectInput("COLLISION" + c, i++, vectLen);

		actions = new IndirectInput[this.actionChoices.size()];
		for (int a = 0; a < actions.length; a++) actions[a] = new IndirectInput("ACTION" + a, i++, vectLen);
		
		usedRels = new ArrayList<IndirectInput>();
		usedRels.add(right);
		usedRels.add(left);
		usedRels.add(above);
		usedRels.add(below);
		usedRels.add(same);
		for (IndirectInput rel : collisions) usedRels.add(rel);
		for (IndirectInput rel : actions) usedRels.add(rel);
		
	}

	/** get input to given output by relation */
	public T getRelNode(T output, IndirectInput rel) {
		return out2InNodeMap.get(output).get(rel);
	}
	
	public IndirectInput getRel(T inNode, T outNode) {
		return relationMap.get(new NodePair<T>(inNode, outNode));
	}
	
	private static double[] geoToVector(int c, int numC, int r, int numR, int o, int numO) {
		// TODO this is not the right vecor length anymore
		double[] result = new double[2 + numO];
		result[0] = ((double)c) / numC;
		result[1] = ((double)r) / numR;
		result[2 + o] = 1;
		return result;
	}
	
	private static <T> void mapNodesToGeoSpace(Map<T, Integer> cols, Map<T, Integer> rows,
			Map<T, Integer> objClass, ArrayList<? extends T> nodes, int coln, int rown) {
		int classI = 0;
		int c = 0;
		int r = 0;
		for (T n : nodes) {
			cols.put(n, c);
			rows.put(n, r);
			objClass.put(n, classI);
			c = (c+1)%coln;
			if (c == 0) {
				r = (r+1)%rown;
				if (r == 0) classI++;
			}
		}
	}

	public Collection<IndirectInput> getUsedRels() {
		return usedRels;
	}
}
