import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import jupyter_drawing_pad as jd
    

def digit_classifier_widget(scaler, apply_classifier):
    
    jdp = jd.CustomBox()
    draw_pad = jdp.drawing_pad
    clear_btn = jdp.children[1].children[1]

    out = widgets.Output(layout=widgets.Layout(width='400px'))

    @out.capture() 
    def w_CB(change):
        from scipy.signal import convolve2d
        from cv2 import resize, INTER_CUBIC, cvtColor, COLOR_RGB2GRAY

        data = change['new']
        if len(data[0]) > 2:
            # Get strokes information
            x = np.array(data[0])
            y = np.array(data[1])
            t = np.array(data[2])

            # assuming there is at least 200ms between each stroke 
            line_breaks = np.where(np.diff(t) > 200)[0]
            # adding end of array
            line_breaks = np.append(line_breaks, t.shape[0])

            # Plot to canvas
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            fig = plt.figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            # plot all strokes
            plt.plot(x[:line_breaks[0]], y[:line_breaks[0]], color='black', linewidth=4)
            for i in range(1, len(line_breaks)):
                plt.plot(x[line_breaks[i-1]+1 : line_breaks[i]], y[line_breaks[i-1]+1 : line_breaks[i]], color='black', linewidth=4)

            plt.xlim(0,460)
            plt.ylim(0,250)
            plt.axis("off")

            canvas.draw()       # draw the canvas, cache the renderer

            # convert to numpy array 
            imageflat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            # not sure why this size...
            image = np.reshape(imageflat,(288, 432, 3))

            # Cut the part containing the writting
            ind = np.where(image<255)      

            D0 = ind[0].max() - ind[0].min() 
            D1 = ind[1].max() - ind[1].min() 

            C0 = int(0.5 * (ind[0].max() + ind[0].min()))
            C1 = int(0.5 * (ind[1].max() + ind[1].min()))

            if D0 > D1:
                D = D0
            else:
                D = D1

            L = int(D / 2.0) + 20
            image = image[C0 - L : C0 + L ,  C1 - L : C1 + L, :]

            # Convert to gray
            image = 255 - cvtColor(image, COLOR_RGB2GRAY)

            # Low pass filter and resize
            k = 12
            I = convolve2d(image, np.ones((k,k))/k**2.0, mode="same")      

            # Resize with opencv 
            I = resize(I, dsize=(28, 28), interpolation=INTER_CUBIC)

            # Clip in [0, 1]
            I = I / I.max()
            I = I * 3.0
            I = I.clip(0, 1)

            # Get a feature vector
            X =  I.reshape((1, 28*28)).astype(np.float64) 

            # Standardization
            X = scaler.transform(X)

            # Apply the classifier
            y_prediction = apply_classifier(X)

            #title = "Prediction: {} ({:.02f})".format(y_prediction, v)    
            title = "Prediction: {}".format(y_prediction)

            # draw the converted image
            plt.clf()
            plt.imshow(I, aspect='equal', cmap = mpl.cm.binary, interpolation='none')
            plt.title(title)
            plt.axis("off")
            plt.show()

            # To erase after tracing
            #change['owner'].data = [[], [], []]

            # Schedule for clearing
            out.clear_output(wait=True)
        else:
            pass

    draw_pad.observe(w_CB, names='data')

    hb = widgets.HBox([draw_pad, clear_btn, out])
    display(hb)