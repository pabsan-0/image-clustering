

imagelinks:
	rm samples/* || true
	ln -s /home/pablo/datasets/robot2020_prefix/robot2020/images/*/*.jpg samples/

clean:
	rm samples/*