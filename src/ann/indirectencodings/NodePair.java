package ann.indirectencodings;

import ann.Node;

public class NodePair {
	
	private Node n1;
	private Node n2;

	public NodePair(Node n1, Node n2) {
		this.n1 = n1;
		this.n2 = n2;
		
	}
	
	@Override
	public int hashCode() {
		return 31 * n1.hashCode() + n2.hashCode();
	}
	
	@Override
	public boolean equals(Object other) {
		if (this == other) return true;
		if (other instanceof NodePair) {
			NodePair onode = (NodePair) other;
			return this.n1.equals(onode.n1) && this.n2.equals(onode.n2);
		}
		return false;
	}
}
