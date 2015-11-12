package modulemanagement;

public class OutputScore<T> implements Comparable<OutputScore<T>> {
	
	T output;
	double score;
	int key;

	public OutputScore(T output, double score, int key) {
		this.output = output;
		this.score = score;
		this.key = key;
	}

	@Override
	public int compareTo(OutputScore<T> o) {
		return Double.compare(this.score, o.score);
	}
}