from PIL import Image, ImageDraw

# Create a blank image with white background
image = Image.new("RGB", (200, 200), "white")
draw = ImageDraw.Draw(image)

# Draw the apple (a red circle)
draw.ellipse((50, 50, 150, 150), fill="red", outline="black")

# Draw the stem (a brown rectangle)
draw.rectangle((95, 20, 105, 50), fill="brown")

# Draw the leaf (a green ellipse)
draw.ellipse((120, 20, 160, 60), fill="green", outline="black")

# Save the image as a PNG file
image.save("apple_image.png")

print("Apple image saved as apple_image.png")