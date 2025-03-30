    def learning_rate_param(self, items):
        value = self._extract_value(items[0])

        # Handle direct learning rate schedule
        if isinstance(value, dict) and 'type' in value and 'args' in value:
            # This is already a learning rate schedule object from the lr_schedule rule
            pass
        elif isinstance(value, dict) and 'hpo' in value:
            # Track HPO for learning_rate
            self._track_hpo('optimizer', 'learning_rate', value, items[0])
        elif isinstance(value, (int, float)) and value <= 0:
            self.raise_validation_error(f"learning_rate must be positive, got {value}", items[0])
        # Handle string-based learning rate schedule for backward compatibility
        elif isinstance(value, str):
            # Special handling for ExponentialDecay
            if value == 'ExponentialDecay':
                # Create a learning rate schedule for ExponentialDecay
                args = []
                if len(items) > 1:
                    # Try to extract the arguments directly
                    try:
                        # If the next item is a token with parentheses, extract the arguments
                        if hasattr(items[1], 'value') and '(' in items[1].value and ')' in items[1].value:
                            args_str = items[1].value.strip('()')
                            # Split by comma and convert to appropriate types
                            for arg in args_str.split(','):
                                arg = arg.strip()
                                # Try to convert to float if possible
                                try:
                                    args.append(float(arg))
                                except ValueError:
                                    # If not a float, keep as string
                                    args.append(arg)
                        else:
                            # Otherwise use the standard extraction
                            args = self._extract_value(items[1])
                            if not isinstance(args, list):
                                args = [args]
                    except Exception as e:
                        # Fallback to standard extraction
                        args = self._extract_value(items[1])
                        if not isinstance(args, list):
                            args = [args]
                value = {
                    'type': 'ExponentialDecay',
                    'args': args
                }
            # Handle other string-based learning rate schedules
            elif '(' in value and ')' in value:
                schedule_str = value.strip('"\'')
                schedule_type = schedule_str[:schedule_str.index('(')]
                args_str = schedule_str[schedule_str.index('(')+1:schedule_str.rindex(')')]

                # Parse arguments
                args = []
                if args_str:
                    import re
                    # Split by comma and convert to appropriate types
                    for arg in args_str.split(','):
                        arg = arg.strip()
                        # Check if this is an HPO expression
                        if arg.startswith('HPO('):
                            # Extract HPO parameters
                            hpo_match = re.search(r'HPO\(([^(]+)\(', arg)
                            if hpo_match:
                                hpo_type = hpo_match.group(1).strip()
                                if hpo_type == 'range':
                                    params = re.search(r'range\(([^,]+),\s*([^,]+)(?:,\s*step=([^)]+))?\)', arg)
                                    if params:
                                        low, high = float(params.group(1)), float(params.group(2))
                                        step = float(params.group(3)) if params.group(3) else None
                                        hpo_dict = {
                                            'type': 'range',
                                            'low': low,
                                            'high': high
                                        }
                                        if step:
                                            hpo_dict['step'] = step
                                        args.append({'hpo': hpo_dict})
                                elif hpo_type == 'log_range':
                                    params = re.search(r'log_range\(([^,]+),\s*([^)]+)\)', arg)
                                    if params:
                                        low, high = float(params.group(1)), float(params.group(2))
                                        args.append({'hpo': {'type': 'log_range', 'low': low, 'high': high}})
                                elif hpo_type == 'choice':
                                    params = re.search(r'choice\(([^)]+)\)', arg)
                                    if params:
                                        choices = [float(x.strip()) for x in params.group(1).split(',')]
                                        args.append({'hpo': {'type': 'choice', 'values': choices}})
                        else:
                            # Try to convert to float if possible
                            try:
                                args.append(float(arg))
                            except ValueError:
                                # If not a float, keep as string
                                args.append(arg)

                value = {
                    'type': schedule_type,
                    'args': args
                }

        return {'learning_rate': value}
