package ann;

import java.util.*;

public class BiasNode {
	
	

	public static final Node INSTANCE = ((ArrayList<Node>)Layer.createInputLayer(1, Node.BASIC_NODE_FACTORY).getNodes()).get(0);
	private static final Map<Node, Connection> toNodesMap = new HashMap<Node, Connection>();
	private static final String BIASNAME = "BIAS";
	
	static {
		INSTANCE.clamp(1);
		INSTANCE.setName(BIASNAME);
	}
	
	public static void connectToLayer(Layer<? extends Node> outputLayer) {
		for (Node n : outputLayer.getNodes()) connectToNode(n);
	}

	public static void connectToNode(Node node) {
		connectToNode(node, null);
	}
	public static void connectToNode(Node node, AccruingWeight weight) {
		Connection conn = weight == null ? Connection.getOrCreate(INSTANCE, node)
				: Connection.getOrCreate(INSTANCE, node, weight);
		toNodesMap.put(node, conn);
		if (toNodesMap.size() > 2000) {
			System.out.println("MEMORY LEAK WARNING! Clear biases please!");
		}
	}
	
	public static void clearConnections() {
		INSTANCE.getOutputConnections().clear();
		toNodesMap.clear();
	}
	
	public static void disconnectFrom(Node n) {
		for (Iterator<Connection> iter = INSTANCE.getOutputConnections().iterator(); iter.hasNext();) {
			Connection conn = iter.next();
			if (conn.getOutputNode() == n) iter.remove();
		}
	}
	
	public static Connection getBiasConnection(Node n) {
		return toNodesMap.get(n);
	}

	public static boolean isBias(Node hNode) {
		return BIASNAME.equals(hNode.toString());
	}
	
}
