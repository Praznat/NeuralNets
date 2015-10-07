package ann.indirectencodings;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ann.Connection;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;
import ann.testing.GridGame;
import modeler.ModelLearner;

public class RelationManager {
	
	public static final IndirectInput RIGHT = new IndirectInput("RIGHT", 1,0,0,0,0,0,0,0,0,0);
	public static final IndirectInput LEFT = new IndirectInput("LEFT", 0,1,0,0,0,0,0,0,0,0,0);
	public static final IndirectInput ABOVE = new IndirectInput("ABOVE", 0,0,1,0,0,0,0,0,0,0,0);
	public static final IndirectInput BELOW = new IndirectInput("BELOW", 0,0,0,1,0,0,0,0,0,0,0);
	public static final IndirectInput SAME = new IndirectInput("SAME", 0,0,0,0,1,0,0,0,0,0);
	public static final IndirectInput COLLISION = new IndirectInput("COLLISION", 0,0,0,0,0,1,0,0,0,0,0);
	public static final IndirectInput NOTHING = new IndirectInput("NOTHING", 0,0,0,0,0,0,1,0,0,0,0);

	public static final IndirectInput[] ACTIONS = new IndirectInput[] {
			new IndirectInput("lACTION", 0,0,0,0,0,0,0,1,0,0,0),
			new IndirectInput("rACTION", 0,0,0,0,0,0,0,0,1,0,0),
			new IndirectInput("uACTION", 0,0,0,0,0,0,0,0,0,1,0),
			new IndirectInput("dACTION", 0,0,0,0,0,0,0,0,0,0,1)
	};
	private static final int ACTION_CLASS = 2;
	
	private final Map<Node, IndirectInput> nodeMap = new HashMap<Node, IndirectInput>();
	private final Map<NodePair, IndirectInput> relationMap = new HashMap<NodePair, IndirectInput>();
	private final Map<Node, Map<IndirectInput, Node>> out2InNodeMap = new HashMap<Node, Map<IndirectInput, Node>>();
	
	/** 
	 * produces an input vector to the CPPN based on the node and connection characteristics 
	 */
	public double[] encodeConnection(Connection conn) {
		Node inNode = conn.getInputNode();
		Node outNode = conn.getOutputNode();
		IndirectInput inV = nodeMap.get(inNode);
		IndirectInput outV = nodeMap.get(outNode);
		IndirectInput relV = relationMap.get(new NodePair(inNode, outNode));
		if (inV == null || outV == null || relV == null) return new double[] {};
		return Utils.concat(inV.getVector(), outV.getVector(), relV.getVector());
	}
	
	public int getInputSize() {
		return Utils.concat(nodeMap.values().iterator().next().getVector(),
				relationMap.values().iterator().next().getVector()).length;
	}
	
	public static RelationManager createFromGridGamePredictor(GridGame game, ModelLearner modeler) {
		RelationManager result = new RelationManager();
		Map<Node, Integer> cols = new HashMap<Node, Integer>();
		Map<Node, Integer> rows = new HashMap<Node, Integer>();
		Map<Node, Integer> objClass = new HashMap<Node, Integer>();
		FFNeuralNetwork predictor = modeler.getTransitionsModule().getNeuralNetwork();
		ArrayList<? extends Node> inputNodes = predictor.getInputNodes();
		ArrayList<? extends Node> outputNodes = predictor.getOutputNodes();

		mapNodesToGeoSpace(cols, rows, objClass, inputNodes, game.cols, game.rows);
		mapNodesToGeoSpace(cols, rows, objClass, outputNodes, game.cols, game.rows);
		
		int numO = (int) Math.ceil(((double)inputNodes.size()) / (game.cols * game.rows));
		
		for (Node out : outputNodes) {
			int outC = cols.get(out);
			int outR = rows.get(out);
			int outO = objClass.get(out);
			result.nodeMap.put(out, new IndirectInput(outC+","+outR+","+outR,
					geoToVector(outC, game.cols, outR, game.cols, outO, numO)));
			Map<IndirectInput, Node> iiToNode = new HashMap<IndirectInput, Node>();
			result.out2InNodeMap.put(out, iiToNode);
			for (Node in : inputNodes) {
				int inC = cols.get(in);
				int inR = rows.get(in);
				int inO = objClass.get(in);
				NodePair pair = new NodePair(in, out);
				IndirectInput rel = NOTHING; // rel means out is REL of in
				result.nodeMap.put(in, new IndirectInput(inC+","+inR+":"+inO,
						geoToVector(inC, game.cols, inR, game.cols, inO, numO)));
				if (inO == ACTION_CLASS) {
					rel = ACTIONS[inC];
				} else if (inC == outC && inR == outR) {
					if (inO == outO) rel = SAME;
					else rel = COLLISION;
				} else if (inO == outO) {
					if (inC == outC) {
						if (outR == inR + 1) rel = BELOW;
						else if (outR == inR - 1) rel = ABOVE;
					} else if (inR == outR) {
						if (outC == inC + 1) rel = RIGHT;
						else if (outC == inC - 1) rel = LEFT;
					}
				}
				result.relationMap.put(pair, rel);
				if (rel != NOTHING) iiToNode.put(rel, in);
			}
		}
		if (result.nodeMap.isEmpty()) throw new IllegalStateException("Empty Relation Manager Created");
		return result;
	}
	
	public Node getRelNode(Node n, IndirectInput rel) {
		return out2InNodeMap.get(n).get(rel);
	}
	
	private static double[] geoToVector(int c, int numC, int r, int numR, int o, int numO) {
		double[] result = new double[2 + numO];
		result[0] = ((double)c) / numC;
		result[1] = ((double)r) / numR;
		result[2 + o] = 1;
		return result;
	}
	
	private static void mapNodesToGeoSpace(Map<Node, Integer> cols, Map<Node, Integer> rows,
			Map<Node, Integer> objClass, ArrayList<? extends Node> nodes, int coln, int rown) {
		int classI = 0;
		int c = 0;
		int r = 0;
		for (Node n : nodes) {
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
}
