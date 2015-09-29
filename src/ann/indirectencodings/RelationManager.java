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
	
	private final Map<Node, IndirectInput> nodeMap = new HashMap<Node, IndirectInput>();
	private final Map<NodePair, IndirectInput> relationMap = new HashMap<NodePair, IndirectInput>();
	
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

		IndirectInput right = new IndirectInput("right", 1,0,0,0,0,0);
		IndirectInput left = new IndirectInput("left", 0,1,0,0,0,0,0);
		IndirectInput above = new IndirectInput("above", 0,0,1,0,0,0,0);
		IndirectInput below = new IndirectInput("below", 0,0,0,1,0,0,0);
		IndirectInput same = new IndirectInput("same", 0,0,0,0,1,0);
		IndirectInput collision = new IndirectInput("collision", 0,0,0,0,0,1,0);
		IndirectInput nothing = new IndirectInput("nothing", 0,0,0,0,0,0,1);
		
		for (Node in : inputNodes) {
			int inC = cols.get(in);
			int inR = rows.get(in);
			int inO = objClass.get(in);
			result.nodeMap.put(in, new IndirectInput(inC+","+inR+":"+inO,
					geoToVector(inC, game.cols, inR, game.cols, inO, numO)));
			for (Node out : outputNodes) {
				NodePair pair = new NodePair(in, out);
				IndirectInput rel = nothing;
				int outC = cols.get(out);
				int outR = rows.get(out);
				int outO = objClass.get(out);
				result.nodeMap.put(out, new IndirectInput(outC+","+outR+","+outR,
						geoToVector(outC, game.cols, outR, game.cols, outO, numO)));
				if (inC == outC && inR == outR) {
					if (inO == outO) rel = same;
					else rel = collision;
				} else if (inO == outO) {
					if (inC == outC) {
						if (outR == inR + 1) rel = below;
						else if (outR == inR - 1) rel = above;
					} else if (inR == outR) {
						if (outC == inC + 1) rel = right;
						else if (outC == inC - 1) rel = left;
					}
				}
				// TODO handle actions more intelligently
				result.relationMap.put(pair, rel);
			}
		}
		return result;
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
