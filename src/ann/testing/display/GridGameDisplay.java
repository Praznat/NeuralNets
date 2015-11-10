package ann.testing.display;

import java.awt.GridLayout;

import javax.swing.JFrame;

public class GridGameDisplay {
	public static JFrame frame = new JFrame();
	public GridPanel gridPanel = new GridPanel(frame, 500);
	public GridGameControlPanel controlPanel = new GridGameControlPanel(frame);
	public GridGameDisplay() {
		frame.setLayout(new GridLayout(0,2));
		frame.add(gridPanel);
		frame.add(controlPanel);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
	}
}
