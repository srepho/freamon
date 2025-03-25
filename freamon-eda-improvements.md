# Freamon EDA Package Improvements

This document outlines recommended modifications to fix issues with report generation and interactive HTML elements in the Freamon EDA package.

## Current Issues

1. **Accordion and Dropdown Functionality**: HTML interactive elements aren't responding to clicks correctly
2. **Report Generation**: HTML layout and JavaScript initialization issues
3. **Large Report Files**: Base64-encoded plots increase file size
4. **Error Handling**: Some errors may be silently caught without proper reporting
5. **Code Organization**: Large monolithic HTML generation functions

## Recommended Solutions

### 1. Fix Accordion Functionality in `report.py`

Replace the current JavaScript section at the end of your `generate_html_report` function with:

```javascript
// Replace your current script section with this:
html += """
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Simple initialization for Bootstrap components
            document.addEventListener('DOMContentLoaded', function() {
                // Initialize tooltips if any
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                tooltipTriggerList.forEach(function(tooltipTriggerEl) {
                    new bootstrap.Tooltip(tooltipTriggerEl);
                });
                
                // Properly handle accordion item clicks
                document.querySelectorAll('.accordion-button').forEach(function(button) {
                    button.addEventListener('click', function(e) {
                        // Prevent default to avoid any conflicts
                        e.preventDefault();
                        
                        // Use Bootstrap's collapse API properly
                        var targetId = this.getAttribute('data-bs-target');
                        var targetCollapse = document.querySelector(targetId);
                        
                        if (targetCollapse) {
                            var bsCollapse = bootstrap.Collapse.getInstance(targetCollapse);
                            
                            // If instance doesn't exist yet, create it
                            if (!bsCollapse) {
                                bsCollapse = new bootstrap.Collapse(targetCollapse, {
                                    toggle: false
                                });
                            }
                            
                            // Toggle the collapse
                            if (targetCollapse.classList.contains('show')) {
                                bsCollapse.hide();
                            } else {
                                bsCollapse.show();
                            }
                        }
                    });
                });
                
                console.log('Freamon EDA report initialized successfully');
            });
        </script>
    </body>
    </html>
    """
```

This script is more compatible with Bootstrap 5 and correctly initializes accordions.

### 2. Improve HTML Structure for Accordions

Ensure proper structure for Bootstrap 5 accordions:

```python
# In your univariate analysis section where you generate accordion items:
html += f"""
<div class="accordion-item">
    <h2 class="accordion-header" id="heading-{col.replace(' ', '_')}">
        <button class="accordion-button collapsed" type="button" 
                data-bs-toggle="collapse" 
                data-bs-target="#collapse-{col.replace(' ', '_')}" 
                aria-expanded="false" 
                aria-controls="collapse-{col.replace(' ', '_')}">
            {col}
        </button>
    </h2>
    <div id="collapse-{col.replace(' ', '_')}" 
         class="accordion-collapse collapse" 
         aria-labelledby="heading-{col.replace(' ', '_')}">
        <div class="accordion-body">
            <!-- Your content here -->
        </div>
    </div>
</div>
"""
```

Key improvements:
- Sanitize IDs by replacing spaces and special characters
- Ensure consistent and correct class names and data attributes
- Proper nesting of HTML elements

### 3. Modular Approach to HTML Generation

Refactor the monolithic HTML generation into modular components:

```python
def generate_html_report(df, analysis_results, output_path, title="Exploratory Data Analysis Report", theme="cosmo"):
    html_parts = []
    
    # Add header
    html_parts.append(generate_header(title, theme))
    
    # Add overview section
    html_parts.append(generate_overview_section(df, analysis_results))
    
    # Add univariate section if available
    if "univariate" in analysis_results:
        html_parts.append(generate_univariate_section(analysis_results["univariate"]))
    
    # Add other sections...
    
    # Add footer with scripts
    html_parts.append(generate_footer())
    
    # Join all parts
    html = "\n".join(html_parts)
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Report saved to {output_path}")
```

Create separate helper functions for each report section:

```python
def generate_header(title, theme):
    """Generate HTML header with proper CSS links"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootswatch@5.2.3/dist/{theme}/bootstrap.min.css">
        <style>
            body {{ padding-top: 20px; padding-bottom: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .card {{ margin-bottom: 20px; }}
            .table-responsive {{ margin-bottom: 20px; }}
            .plot-img {{ max-width: 100%; height: auto; }}
            .nav-pills .nav-link.active {{ background-color: #6c757d; }}
            
            /* Accordion styles */
            .accordion-button:not(.collapsed) {{
                background-color: #e7f1ff;
                color: #0c63e4;
                box-shadow: inset 0 -1px 0 rgba(0,0,0,.125);
            }}
            .accordion-button.collapsed {{
                background-color: #f8f9fa;
            }}
            .accordion-item {{
                border: 1px solid rgba(0,0,0,.125);
                margin-bottom: 5px;
            }}
            .accordion-body {{
                padding: 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">{title}</h1>
            <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

def generate_footer():
    """Generate HTML footer with proper scripts"""
    return """
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Simple initialization for Bootstrap components
            document.addEventListener('DOMContentLoaded', function() {
                // Initialize tooltips if any
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                tooltipTriggerList.forEach(function(tooltipTriggerEl) {
                    new bootstrap.Tooltip(tooltipTriggerEl);
                });
                
                // Properly handle accordion item clicks
                document.querySelectorAll('.accordion-button').forEach(function(button) {
                    button.addEventListener('click', function(e) {
                        // Prevent default to avoid any conflicts
                        e.preventDefault();
                        
                        // Use Bootstrap's collapse API properly
                        var targetId = this.getAttribute('data-bs-target');
                        var targetCollapse = document.querySelector(targetId);
                        
                        if (targetCollapse) {
                            var bsCollapse = bootstrap.Collapse.getInstance(targetCollapse);
                            
                            // If instance doesn't exist yet, create it
                            if (!bsCollapse) {
                                bsCollapse = new bootstrap.Collapse(targetCollapse, {
                                    toggle: false
                                });
                            }
                            
                            // Toggle the collapse
                            if (targetCollapse.classList.contains('show')) {
                                bsCollapse.hide();
                            } else {
                                bsCollapse.show();
                            }
                        }
                    });
                });
                
                console.log('Freamon EDA report initialized successfully');
            });
        </script>
    </body>
    </html>
    """
```

### 4. Reduce Base64 Image Size

Add a function to optimize images before including them in the report:

```python
def optimize_base64_image(base64_str, max_size=800):
    """Resize a base64 image if it's too large"""
    # Extract the base64 data
    if ',' in base64_str:
        header, data = base64_str.split(',', 1)
    else:
        header = "data:image/png;base64"
        data = base64_str
    
    # Decode base64
    binary_data = base64.b64decode(data)
    
    # Open image
    from PIL import Image
    from io import BytesIO
    
    img = Image.open(BytesIO(binary_data))
    
    # Check if resizing is needed
    if max(img.size) > max_size:
        # Calculate new dimensions
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Resize image
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Save to buffer with compression
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True, quality=85)
    buffer.seek(0)
    
    # Encode to base64
    new_data = base64.b64encode(buffer.read()).decode("utf-8")
    
    return f"{header},{new_data}" if ',' in base64_str else new_data
```

Use this function when adding plots to your report:

```python
# Before adding a plot to the report
if "plot" in result:
    optimized_plot = optimize_base64_image(result["plot"])
    html += f"""
        <div class="col-md-6">
            <img src="{optimized_plot}" class="plot-img" alt="Distribution of {col}">
        </div>
    """
```

### 5. Improve Error Handling

Make errors more visible in your reports:

```python
# Add this to your error handling sections
if "error" in results:
    error_message = results["error"]
    html += f"""
    <div class="alert alert-danger" role="alert">
        <strong>Error:</strong> {error_message}
    </div>
    """
```

Also improve error logging:

```python
# Add this to your error handling in analysis functions
except Exception as e:
    logger.error(f"Error in analyzing {column}: {str(e)}")
    return {"error": f"Analysis failed: {str(e)}"}
```

### 6. Additional Bootstrap 5 Compatibility Improvements

1. Update the tab structure to ensure proper Bootstrap 5 compatibility:

```python
# When generating tabs
html += """
<ul class="nav nav-pills mb-4" id="eda-tabs" role="tablist">
    <li class="nav-item" role="presentation">
        <button class="nav-link active" id="overview-tab" data-bs-toggle="pill" 
                data-bs-target="#overview" type="button" role="tab" 
                aria-controls="overview" aria-selected="true">Overview</button>
    </li>
    <!-- Additional tabs -->
</ul>
"""
```

2. Use `data-bs-` prefix consistently for all Bootstrap 5 data attributes.

## Implementation Strategy

1. **Start with JavaScript Fixes**: Begin by fixing the accordion initialization code, as this is likely the source of most issues.

2. **Test in Jupyter**: Test the changes in a Jupyter environment to ensure they work correctly.

3. **Implement Modular Functions**: Gradually refactor the code to use a modular approach, starting with the most problematic sections.

4. **Optimize Images**: Add image optimization to reduce report file sizes.

5. **Improve Error Handling**: Enhance error reporting throughout the package.

These changes should significantly improve your report generation while keeping it compatible with Jupyter environments.
