# Function to check for errors and exit if the previous command failed
function Test-Error {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

# Cleanup function to delete auxiliary files
function Reset-All {
    Remove-Item *.pdf, *.aux, *.bbl, *.bcf, *.blg, *.log, *.toc, *.out, *.run.xml
    # Uncomment the line below for recursive removal
    # Remove-Item -Force -Recurse *.pdf, *.aux, *.bbl, *.bcf, *.blg, *.log, *.toc, *.out, *.run.xml -ErrorAction SilentlyContinue
}

# Function to build LaTeX
function Build-LaTeX {
    # Step 1: Run pdflatex
    Write-Host "Running pdflatex..."
    pdflatex report
    Test-Error

    # Step 2: Run biber
    Write-Host "Running biber..."
    biber report
    Test-Error

    # Step 3: Run pdflatex again
    Write-Host "Running pdflatex again..."
    pdflatex report
    Test-Error
}

# Validate arguments count
if ($args.Count -gt 1) {
    Write-Error "Max 1 parameter accepted."
    exit 1
}

# Handle script arguments using switch
switch ($args[0]) {
    "build" {
        Build-LaTeX
    }
    "clean" {
        Reset-All
    }
    "rebuild" {
        Reset-All
        Build-LaTeX
    }
    Default {
        Write-Host "Usage: script.ps1 {build|clean|rebuild}"
        exit 1
    }
}
