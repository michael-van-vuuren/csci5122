---
title: 
toc: false
---

## CSCI 5122 Semester Project

![Splash](/splash2.png)

{{% details title="Click here for a playable demo!" closed="true" %}}
<div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
    Use the <b>←↑→</b> or <b>AWD keys</b> to land the rocket.
</div>

<div style="display: flex; justify-content: center;">
    <div style="
        position: relative; 
        width: 60%;
        padding-top: 83.6%;
        height: 0;
        border: 3px solid black;
        overflow: hidden;
        margin: 0;
        ">
        <iframe 
            src="/csci5122/web/index.html" 
            style="
                position: absolute; 
                top: 0; 
                left: 0; 
                width: 100%; 
                height: 100%; 
                border: none;
            "
            allow="autoplay; fullscreen; gamepad; microphone; camera"
            tabindex="0"
        ></iframe>
    </div>
</div>

{{% /details %}}

{{< cards >}}
  {{< card link="introduction" title="Introduction" icon="book-open" subtitle="Visit here for an introduction to the project." >}}
  {{< card link="eda" title="Data Prep/EDA" icon="clipboard" subtitle="Visit here to learn about the data." >}}
  {{< card link="models" title="Models" icon="refresh" subtitle="Visit here to learn about neuroevolution." >}}
  {{< card link="results" title="Results" icon="trending-up" subtitle="Visit here for to see the evolved models." >}}
{{< /cards >}}
{{< cards cols="1" >}}
  {{< card link="conclusion" title="Conclusion" icon="check-circle" subtitle="Visit here for the key outcomes and conclusions of the project." >}}
{{< /cards >}}

## This project used the following open source libraries:

1. **NEAT-Python**, for neuroevolution: https://neat-python.readthedocs.io/en/latest/
2. **Pygame**, for simulating the rocket: https://www.pygame.org/docs/
3. **Pillow**, for visualizing the neural networks: https://pillow.readthedocs.io/en/stable/

>[!TIP]
>You can use the tabs to navigate between sections.