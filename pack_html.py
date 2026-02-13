import json

# Input files
html_file = "index.html"      # your HTML file
json_file = "chronoscope.json"  # your JSON file

# Output file
output_file = "standalone.html"

# Read the HTML
with open(html_file, "r", encoding="utf-8") as f:
    html_content = f.read()

# Read the JSON
with open(json_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Convert JSON to a minified string
json_str = json.dumps(json_data)

# Replace the fetch call in HTML with inline JSON
# We'll look for: fetch('chronoscope.json')
# and replace it with: graphsData = <JSON>;
html_content = html_content.replace(
    "fetch(jsonPath)",
    f"Promise.resolve(new Response(JSON.stringify({json_str})))"
)

# Save the standalone HTML
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Standalone HTML saved as {output_file}")
