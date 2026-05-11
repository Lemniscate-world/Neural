# Security Policy

## Supported Versions

We actively maintain security updates for the following versions of NeuralDBG:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in NeuralDBG, please report it to us as follows:

### Contact Information

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:
- **Email**: Lemniscate_zero@proton.me

### What to Include

When reporting a security vulnerability, please include:

1. A clear description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact and severity assessment
4. Any suggested fixes or mitigations
5. Your contact information for follow-up

### Response Timeline

We will acknowledge your report within 48 hours and provide a more detailed response within 7 days indicating our next steps.

We will keep you informed about our progress throughout the process of fixing the vulnerability.

## Security Considerations

### Data Handling

NeuralDBG captures and stores sensitive training data including:

- **Model parameters and weights**
- **Training tensors and gradients**
- **Network architecture information**
- **Training hyperparameters**

### Privacy Considerations

When using NeuralDBG:

- **Research Data**: Be aware that captured traces may contain sensitive information about your models or training data
- **Storage**: Tensor snapshots are stored in memory - ensure adequate system resources
- **Sharing**: Do not share debug traces containing proprietary model information
- **Cleanup**: Clear debug sessions after use to prevent accidental data exposure

### Memory Safety

- NeuralDBG performs tensor cloning and detaching operations
- Memory usage scales with model size and training duration
- Monitor system resources when debugging large models
- Consider disk-based storage for long training sessions

## Security Best Practices

### For Users

1. **Environment Isolation**: Run NeuralDBG in isolated environments
2. **Data Sanitization**: Avoid debugging with sensitive or proprietary data
3. **Version Updates**: Keep NeuralDBG updated to the latest secure version
4. **Resource Monitoring**: Watch memory usage during debugging sessions

### For Contributors

1. **Code Review**: All changes undergo security-focused code review
2. **Dependency Scanning**: Dependencies are regularly scanned for vulnerabilities
3. **Testing**: Security implications are considered in test coverage
4. **Documentation**: Security considerations are documented in code comments

## Vulnerability Classification

We classify vulnerabilities using the following severity levels:

- **Critical**: Immediate threat to user data or system security
- **High**: Significant security risk with potential for exploitation
- **Medium**: Security weakness with limited exploitation potential
- **Low**: Minor security improvements or hardening opportunities

## Security Updates

Security updates will be:

1. Released as patch versions (e.g., 1.0.1, 1.0.2)
2. Documented in release notes with appropriate severity indicators
3. Communicated through our security mailing list
4. Coordinated with downstream package maintainers

## Recognition

We appreciate security researchers who help keep NeuralDBG safe. With your permission, we will publicly acknowledge your contribution in our security advisories and release notes.

## Questions

If you have questions about this security policy or security practices, please contact us.
