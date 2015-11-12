package ann.indirectencodings;

public class NodePair<T> {
	
	private T n1;
	private T n2;

	public NodePair(T n1, T n2) {
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
		if (other instanceof NodePair<?>) {
			@SuppressWarnings("unchecked")
			NodePair<T> onode = (NodePair<T>) other;
			return this.n1.equals(onode.n1) && this.n2.equals(onode.n2);
		}
		return false;
	}
}
