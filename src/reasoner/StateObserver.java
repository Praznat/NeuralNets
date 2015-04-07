package reasoner;

public interface StateObserver {
	public void observeState(int t, double realism, double... state);
}
