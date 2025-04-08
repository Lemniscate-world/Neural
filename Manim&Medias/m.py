from manim import *

class NeuralIntro(Scene):
    def construct(self):
        # Scene 1: Intro
        title = Text("Introducing Neural", font_size=48, color=WHITE)
        subtitle = Text("A DSL and Debugger for Neural Networks", font_size=24, color=GRAY).next_to(title, DOWN)
        self.play(Write(title), Write(subtitle))
        self.wait(1)

        # Simple neural network diagram
        layers = VGroup(*[
            Circle(radius=0.5, fill_opacity=1, fill_color=BLUE).shift(LEFT * 4 + UP * i)
            for i in range(-1, 2)
        ], *[
            Circle(radius=0.5, fill_opacity=1, fill_color=GREEN).shift(RIGHT * 4 + UP * i)
            for i in range(-1, 2)
        ])
        lines = VGroup(*[
            Line(layers[i].get_center(), layers[j].get_center(), color=WHITE)
            for i in range(3) for j in range(3, 6)
        ])
        nn = VGroup(layers, lines).scale(0.5).next_to(subtitle, DOWN, buff=1)
        self.play(FadeIn(nn), FadeOut(title), FadeOut(subtitle))
        self.wait(1)

        # Scene 2: DSL Overview
        dsl_code_text = """network MyNet {
    input: (28, 28, 1)
    layers: Conv2D(filters=16, kernel_size=3)
            Flatten()
            Dense(128)
            Output(10)
}"""
        dsl_code = Code(code=dsl_code_text, tab_width=4, background="window", language="yaml", font_size=24).to_edge(LEFT)
        self.play(FadeOut(nn), Write(dsl_code))
        self.wait(1)

        # Transform code to network diagram
        conv = Rectangle(width=2, height=1, fill_opacity=1, fill_color=BLUE).shift(LEFT * 2).add(Tex("Conv2D").scale(0.5))
        flat = Rectangle(width=2, height=1, fill_opacity=1, fill_color=GREEN).add(Tex("Flatten").scale(0.5))
        dense = Rectangle(width=2, height=1, fill_opacity=1, fill_color=YELLOW).shift(RIGHT * 2).add(Tex("Dense").scale(0.5))
        out = Rectangle(width=2, height=1, fill_opacity=1, fill_color=RED).shift(RIGHT * 4).add(Tex("Output").scale(0.5))
        arrows = VGroup(
            Arrow(conv.get_right(), flat.get_left()),
            Arrow(flat.get_right(), dense.get_left()),
            Arrow(dense.get_right(), out.get_left())
        )
        network = VGroup(conv, flat, dense, out, arrows)
        self.play(Transform(dsl_code, network))
        self.wait(1)

        # Scene 3: Multi-Backend Export
        frameworks = VGroup(
            Text("TensorFlow", font_size=24, color=ORANGE).shift(UP * 2),
            Text("PyTorch", font_size=24, color=RED).shift(UP * 0),
            Text("ONNX", font_size=24, color=GREEN).shift(DOWN * 2)
        ).to_edge(RIGHT)
        self.play(FadeOut(network), Write(frameworks))
        self.wait(1)

        # Scene 4: NeuralDbg
        debug_title = Text("NeuralDbg: Real-Time Debugging", font_size=36).to_edge(UP)
        self.play(FadeOut(frameworks), Write(debug_title))
        
        # Network with flowing activations
        nn_copy = nn.copy().shift(DOWN * 1.5)
        flow = VGroup(*[
            Dot(color=YELLOW).move_to(layers[i]).animate.move_to(layers[j + 3])
            for i in range(3) for j in range(3)
        ])
        self.play(FadeIn(nn_copy), *flow)
        
        # 3D diagram placeholder (simplified as a cube)
        cube = Cube(fill_opacity=0.5, fill_color=BLUE).scale(0.5).shift(RIGHT * 3)
        self.play(Create(cube))
        self.wait(1)

        # Scene 5: Conclusion
        benefits = BulletedList(
            "Declarative DSL",
            "Shape Propagation",
            "Multi-Backend Support",
            "Real-Time Debugging",
            font_size=24
        ).to_edge(LEFT)
        self.play(FadeOut(nn_copy), FadeOut(cube), FadeOut(debug_title), Write(benefits))
        self.wait(1)
        
        cta = Text("Try Neural Today!", font_size=36, color=YELLOW).next_to(benefits, DOWN)
        self.play(Write(cta))
        self.wait(2)