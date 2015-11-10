package ann.testing.display;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import ann.testing.GridExploreGame;
import ann.testing.GridGame;
import ann.testing.GridTagGame;

public class GridGameControlPanel extends JPanel implements ActionListener {
	JFrame parent;
	GridGame game;
	JButton bL = new JButton("Predict-L");
	JButton bR = new JButton("Predict-R");
	JButton bU = new JButton("Predict-U");
	JButton bD = new JButton("Predict-D");
	JButton bRestore = new JButton("Predict-Restore");
	JButton b1 = new JButton("Catch");
	JButton b2 = new JButton("Evade");
	private long lastClickMs;
	private double[] savedState;
	public GridGameControlPanel(JFrame parent) {
		this.parent = parent;
		this.add(bL);
		this.add(bR);
		this.add(bU);
		this.add(bD);
		this.add(bRestore);
		this.add(b1);
		this.add(b2);
		bL.addActionListener(this);
		bR.addActionListener(this);
		bU.addActionListener(this);
		bD.addActionListener(this);
		bRestore.addActionListener(this);
		b1.addActionListener(this);
		b2.addActionListener(this);
	}
	public void setGame(GridGame game) {
		this.game = game;
		parent.pack();
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		final String command = e.getActionCommand();
		long ms = System.currentTimeMillis();
		if (ms < lastClickMs + 50) return;
		lastClickMs = ms;
		if (command.startsWith("Predict")) {
			int joints = 3;
			if (command.equals("Predict-Restore") && savedState != null) {
				game.convertFromState(savedState);
			} else {
				double[] state = game.getState();
				if (savedState == null) savedState = state;
				double[] action = null;
				if (command.equals("Predict-L")) {
					action = GridExploreGame.LEFT;
				} else if (command.equals("Predict-R")) {
					action = GridExploreGame.RIGHT;
				} else if (command.equals("Predict-U")) {
					action = GridExploreGame.UP;
				} else if (command.equals("Predict-D")) {
					action = GridExploreGame.DOWN;
				}
				System.out.println(game.modeler);
				double[] newStateVars = game.modeler.newStateVars(state, action, joints);
				//			System.out.println(Utils.stringArray(newStateVars, 2));
				game.convertFromState(newStateVars);
			}
			parent.repaint();
//			game.convertFromState(state);
//			repaint();
		}
		else if (command.equals("Catch")) {
			game.setRewardFunction(GridTagGame.follow(game));
		}
		else if (command.equals("Evade")) {
			game.setRewardFunction(GridTagGame.evade(game));
		}
	}
}
